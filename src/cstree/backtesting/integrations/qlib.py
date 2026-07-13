"""Lazy Qlib backtest adapter with a framework-neutral output boundary."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import import_module
from typing import Any

import pandas as pd

from ..backends import BacktestBackend, BacktestBackendResult

QlibRunner = Callable[..., object]


class QlibUnavailableError(RuntimeError):
    """Raised when the optional Qlib backend is used without pyqlib installed."""


@dataclass(frozen=True)
class QlibBacktestRequest:
    """Serializable arguments accepted by the official Qlib backtest function."""

    start_time: str | pd.Timestamp
    end_time: str | pd.Timestamp
    strategy: str | Mapping[str, Any]
    executor: str | Mapping[str, Any]
    benchmark: str = "SH000300"
    account: float | int | Mapping[str, Any] = 1_000_000_000.0
    exchange_kwargs: Mapping[str, Any] | None = None
    pos_type: str = "Position"
    frequency: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.strategy, (str, Mapping)):
            raise TypeError("strategy must be a string or a serializable Qlib config mapping.")
        if not isinstance(self.executor, (str, Mapping)):
            raise TypeError("executor must be a string or a serializable Qlib config mapping.")
        if not isinstance(self.account, (int, float, Mapping)):
            raise TypeError("account must be numeric or a serializable config mapping.")
        if not self.benchmark.strip() or not self.pos_type.strip():
            raise ValueError("benchmark and pos_type must not be empty.")
        object.__setattr__(self, "strategy", _copy_config(self.strategy))
        object.__setattr__(self, "executor", _copy_config(self.executor))
        object.__setattr__(self, "account", _copy_config(self.account))
        object.__setattr__(self, "exchange_kwargs", _copy_config(self.exchange_kwargs or {}))


class QlibBacktestBackend(BacktestBackend[QlibBacktestRequest]):
    """Invoke ``qlib.backtest.backtest`` and normalize its portfolio metrics."""

    def __init__(self, *, runner: QlibRunner | None = None) -> None:
        self._runner = runner

    @property
    def name(self) -> str:
        return "qlib-backtest"

    def run(self, request: QlibBacktestRequest) -> BacktestBackendResult:
        runner = self._runner or _load_official_qlib_backtest()
        raw_result = runner(
            start_time=request.start_time,
            end_time=request.end_time,
            strategy=_copy_config(request.strategy),
            executor=_copy_config(request.executor),
            benchmark=request.benchmark,
            account=_copy_config(request.account),
            exchange_kwargs=_copy_config(request.exchange_kwargs or {}),
            pos_type=request.pos_type,
        )
        portfolio_metrics, indicator_metrics = _split_qlib_result(raw_result)
        frequency, report, positions = _select_frequency(
            portfolio_metrics,
            requested=request.frequency,
        )
        performance = _normalize_qlib_report(report)
        canonical_positions = _normalize_qlib_positions(positions)
        metadata = {
            "canonical": False,
            "frequency": frequency,
            "benchmark": request.benchmark,
            "indicator_frequencies": sorted(str(key) for key in indicator_metrics),
            "cost_classification": "qlib_cost_reported_as_fee_cost",
            "position_semantics": "post_trade_account",
            "pnl_semantics": "compounded_net_return",
        }
        return BacktestBackendResult(self.name, performance, canonical_positions, metadata)


def _copy_config(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Qlib config values must be finite.")
        return value
    if isinstance(value, Mapping):
        copied: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("Qlib config mapping keys must be strings.")
            copied[key] = _copy_config(item)
        return copied
    if isinstance(value, (list, tuple)):
        return [_copy_config(item) for item in value]
    raise TypeError(
        f"Qlib configs must contain JSON-compatible values; got {type(value).__name__}."
    )


def _load_official_qlib_backtest() -> QlibRunner:
    try:
        module = import_module("qlib.backtest")
    except ModuleNotFoundError as exc:
        if exc.name == "qlib" or (exc.name or "").startswith("qlib."):
            raise QlibUnavailableError(
                "The Qlib backend requires the optional 'qlib' extra. "
                "Install portfolio-backtester[qlib] before running it."
            ) from exc
        raise
    runner = getattr(module, "backtest", None)
    if not callable(runner):
        raise QlibUnavailableError("Installed pyqlib does not expose qlib.backtest.backtest.")
    return runner


def _split_qlib_result(raw_result: object) -> tuple[Mapping[Any, Any], Mapping[Any, Any]]:
    if not isinstance(raw_result, tuple) or len(raw_result) != 2:
        raise TypeError("Qlib backtest must return (portfolio_metrics, indicator_metrics).")
    portfolio_metrics, indicator_metrics = raw_result
    if not isinstance(portfolio_metrics, Mapping) or not isinstance(indicator_metrics, Mapping):
        raise TypeError("Qlib backtest outputs must be mappings keyed by frequency.")
    return portfolio_metrics, indicator_metrics


def _select_frequency(
    portfolio_metrics: Mapping[Any, Any],
    *,
    requested: str | None,
) -> tuple[str, pd.DataFrame, Mapping[Any, Any]]:
    if not portfolio_metrics:
        raise ValueError("Qlib backtest returned no portfolio metrics.")
    by_name = {str(key): value for key, value in portfolio_metrics.items()}
    if requested is None:
        if len(by_name) != 1:
            available = ", ".join(sorted(by_name))
            raise ValueError(
                "Qlib returned multiple frequencies; set request.frequency. "
                f"Available frequencies: {available}."
            )
        frequency = next(iter(by_name))
    else:
        frequency = requested
        if frequency not in by_name:
            available = ", ".join(sorted(by_name))
            raise ValueError(
                f"Qlib frequency {frequency!r} is unavailable. Available: {available}."
            )
    payload = by_name[frequency]
    if not isinstance(payload, tuple) or len(payload) != 2:
        raise TypeError("Each Qlib portfolio metric entry must be (report, positions).")
    report, positions = payload
    if not isinstance(report, pd.DataFrame) or not isinstance(positions, Mapping):
        raise TypeError("Qlib portfolio metrics require a DataFrame report and positions mapping.")
    return frequency, report, positions


def _normalize_qlib_report(report: pd.DataFrame) -> pd.DataFrame:
    required = {"return", "turnover", "cost"}
    missing = sorted(required - set(report.columns))
    if missing:
        raise ValueError("Qlib portfolio report is missing column(s): " + ", ".join(missing))
    gross_return = pd.to_numeric(report["return"], errors="raise").astype(float)
    total_cost = pd.to_numeric(report["cost"], errors="raise").astype(float)
    net_return = gross_return - total_cost
    return pd.DataFrame(
        {
            "date": pd.to_datetime(report.index, errors="raise"),
            "gross_return": gross_return.to_numpy(),
            "net_return": net_return.to_numpy(),
            "turnover": pd.to_numeric(report["turnover"], errors="raise").to_numpy(),
            "fee_cost": total_cost.to_numpy(),
            "slippage_cost": 0.0,
            "total_cost": total_cost.to_numpy(),
            "pnl": ((1.0 + net_return).cumprod() - 1.0).to_numpy(),
        }
    )


def _normalize_qlib_positions(positions: Mapping[Any, Any]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for date, snapshot in positions.items():
        weights = _extract_weight_mapping(snapshot)
        for symbol, weight in weights.items():
            records.append({"date": date, "symbol": str(symbol), "weight": float(weight)})
    if not records:
        return pd.DataFrame({"date": [], "symbol": [], "weight": []})
    return pd.DataFrame(records)


def _extract_weight_mapping(snapshot: Any) -> Mapping[Any, Any]:
    getter = getattr(snapshot, "get_stock_weight_dict", None)
    if callable(getter):
        weights = getter()
        if not isinstance(weights, Mapping):
            raise TypeError("Qlib position get_stock_weight_dict() must return a mapping.")
        return weights
    if isinstance(snapshot, Mapping):
        return {
            symbol: weight
            for symbol, weight in snapshot.items()
            if symbol != "cash" and isinstance(weight, (int, float))
        }
    raise TypeError("Unsupported Qlib position snapshot; expected Position or weight mapping.")


__all__ = ["QlibBacktestBackend", "QlibBacktestRequest", "QlibUnavailableError"]
