"""Framework-neutral backtest backend contracts and the canonical native backend."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar, cast

import numpy as np
import pandas as pd

from .position_backtest import (
    PositionBacktestConfig,
    normalize_position_backtest_positions,
    run_position_backtest,
)

PERFORMANCE_COLUMNS = (
    "date",
    "gross_return",
    "net_return",
    "turnover",
    "fee_cost",
    "slippage_cost",
    "total_cost",
    "pnl",
)
POSITION_COLUMNS = ("date", "symbol", "weight")

RequestT_contra = TypeVar("RequestT_contra", contravariant=True)


class BacktestBackend(Protocol[RequestT_contra]):
    """Port implemented by native and optional framework-specific backends."""

    @property
    def name(self) -> str:
        """Return the stable backend identifier."""

        ...

    def run(self, request: RequestT_contra) -> BacktestBackendResult:
        """Run a backtest and return the framework-neutral result."""

        ...


@dataclass(frozen=True)
class BacktestBackendResult:
    """Canonical result used for persistence and differential comparisons."""

    backend: str
    performance: pd.DataFrame
    positions: pd.DataFrame
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.backend.strip():
            raise ValueError("backend must not be empty.")
        performance = _normalize_performance(self.performance)
        positions = _normalize_positions(self.positions)
        object.__setattr__(self, "performance", performance)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "metadata", _neutral_metadata(self.metadata))


@dataclass(frozen=True)
class PositionReplayRequest:
    """Inputs for the deterministic native position replay backend."""

    positions: pd.DataFrame
    pricing: pd.DataFrame
    periods: pd.DataFrame
    config: PositionBacktestConfig
    intraday_bars: pd.DataFrame | None = None


class NativeAShareReplayBackend(BacktestBackend[PositionReplayRequest]):
    """Canonical deterministic replay for A-share execution and cost semantics."""

    @property
    def name(self) -> str:
        return "native-a-share-replay"

    def run(self, request: PositionReplayRequest) -> BacktestBackendResult:
        result = run_position_backtest(
            positions=request.positions,
            pricing=request.pricing,
            periods=request.periods,
            config=request.config,
            intraday_bars=request.intraday_bars,
        )
        periods = result.periods
        net_returns = pd.to_numeric(periods["net_return"], errors="raise").astype(float)
        performance = pd.DataFrame(
            {
                "date": pd.to_datetime(periods["exit_date"], errors="raise"),
                "gross_return": periods["gross_return"],
                "net_return": net_returns,
                "turnover": periods["turnover"],
                "fee_cost": periods["fee_cost"],
                "slippage_cost": periods["slippage_cost"],
                "total_cost": periods["total_cost"],
                "pnl": (1.0 + net_returns).cumprod() - 1.0,
            }
        )
        normalized_positions = normalize_position_backtest_positions(request.positions)
        positions = normalized_positions.rename(columns={"rebalance_key": "date"})[
            ["date", "symbol", "weight"]
        ]
        metadata = {
            "canonical": True,
            "engine_schema": result.summary.get("schema"),
            "price_col": request.config.price_col,
            "exit_price_policy": request.config.exit_price_policy,
            "position_semantics": "target_rebalance",
            "pnl_semantics": "compounded_net_return",
        }
        return BacktestBackendResult(self.name, performance, positions, metadata)


def _normalize_performance(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(PERFORMANCE_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError("Backtest performance is missing column(s): " + ", ".join(missing))
    out = frame.loc[:, PERFORMANCE_COLUMNS].copy()
    out["date"] = pd.to_datetime(out["date"], errors="raise")
    if out["date"].isna().any():
        raise ValueError("Backtest performance dates must not be missing.")
    for column in PERFORMANCE_COLUMNS[1:]:
        out[column] = pd.to_numeric(out[column], errors="raise").astype(float)
        if not np.isfinite(out[column]).all():
            raise ValueError(f"Backtest performance column {column!r} contains non-finite values.")
    if out["date"].duplicated().any():
        raise ValueError("Backtest performance contains duplicate dates.")
    return out.sort_values("date", kind="stable").reset_index(drop=True)


def _normalize_positions(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(POSITION_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError("Backtest positions are missing column(s): " + ", ".join(missing))
    out = frame.loc[:, POSITION_COLUMNS].copy()
    out["date"] = pd.to_datetime(out["date"], errors="raise")
    if out["date"].isna().any():
        raise ValueError("Backtest position dates must not be missing.")
    out["symbol"] = out["symbol"].astype(str)
    out["weight"] = pd.to_numeric(out["weight"], errors="raise").astype(float)
    if not np.isfinite(out["weight"]).all():
        raise ValueError("Backtest position weights contain non-finite values.")
    if out.duplicated(["date", "symbol"]).any():
        raise ValueError("Backtest positions contain duplicate date and symbol rows.")
    return out.sort_values(["date", "symbol"], kind="stable").reset_index(drop=True)


def _neutral_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    """Copy metadata while rejecting framework/runtime objects."""

    def normalize(item: Any, path: str) -> Any:
        if item is None or isinstance(item, (str, bool, int)):
            return item
        if isinstance(item, float):
            if not np.isfinite(item):
                raise ValueError(f"Backtest metadata {path} must be finite.")
            return item
        if isinstance(item, Mapping):
            source = cast(Mapping[Any, Any], item)
            normalized: dict[str, Any] = {}
            for key, nested in source.items():
                if not isinstance(key, str):
                    raise TypeError(f"Backtest metadata {path} keys must be strings.")
                normalized[key] = normalize(nested, f"{path}.{key}")
            return normalized
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            return [normalize(nested, f"{path}[]") for nested in item]
        raise TypeError(
            f"Backtest metadata {path} contains non-neutral value {type(item).__name__}."
        )

    return cast(dict[str, Any], normalize(value, "metadata"))


__all__ = [
    "BacktestBackend",
    "BacktestBackendResult",
    "NativeAShareReplayBackend",
    "PositionReplayRequest",
]
