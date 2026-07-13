from __future__ import annotations

import math
from typing import Any

import pandas as pd
import pytest

from cstree.backtesting.integrations import qlib as qlib_adapter
from cstree.backtesting.integrations.qlib import (
    QlibBacktestBackend,
    QlibBacktestRequest,
    QlibUnavailableError,
)


class _FakeQlibPosition:
    def __init__(self, weights: dict[str, float]) -> None:
        self._weights = weights

    def get_stock_weight_dict(self) -> dict[str, float]:
        return dict(self._weights)


def _qlib_output() -> tuple[dict[str, Any], dict[str, Any]]:
    report = pd.DataFrame(
        {
            "return": [0.01, 0.02],
            "turnover": [0.3, 0.4],
            "cost": [0.001, 0.002],
            "account": [1_009_000.0, 1_027_162.0],
        },
        index=pd.to_datetime(["2020-01-02", "2020-01-03"]),
    )
    positions = {
        pd.Timestamp("2020-01-02"): _FakeQlibPosition({"AAA": 0.6, "BBB": 0.4}),
        pd.Timestamp("2020-01-03"): _FakeQlibPosition({"AAA": 1.0}),
    }
    return {"1day": (report, positions)}, {"1day": (pd.DataFrame(), object())}


def test_qlib_backend_calls_runner_with_official_signature_and_normalizes_output() -> None:
    calls: list[dict[str, Any]] = []

    def runner(**kwargs: Any) -> object:
        calls.append(kwargs)
        return _qlib_output()

    request = QlibBacktestRequest(
        start_time="2020-01-01",
        end_time="2020-01-03",
        strategy={"class": "TopkDropoutStrategy", "kwargs": {"topk": 20}},
        executor={"class": "SimulatorExecutor", "kwargs": {"time_per_step": "day"}},
        account=1_000_000.0,
        exchange_kwargs={"open_cost": 0.0005},
    )

    result = QlibBacktestBackend(runner=runner).run(request)

    assert calls == [
        {
            "start_time": "2020-01-01",
            "end_time": "2020-01-03",
            "strategy": {"class": "TopkDropoutStrategy", "kwargs": {"topk": 20}},
            "executor": {"class": "SimulatorExecutor", "kwargs": {"time_per_step": "day"}},
            "benchmark": "SH000300",
            "account": 1_000_000.0,
            "exchange_kwargs": {"open_cost": 0.0005},
            "pos_type": "Position",
        }
    ]
    assert result.metadata["frequency"] == "1day"
    assert result.metadata["position_semantics"] == "post_trade_account"
    assert result.performance["net_return"].tolist() == pytest.approx([0.009, 0.018])
    assert math.isclose(result.performance.iloc[-1]["pnl"], 0.027162, abs_tol=1e-12)
    assert result.positions.to_dict("records")[-1] == {
        "date": pd.Timestamp("2020-01-03"),
        "symbol": "AAA",
        "weight": 1.0,
    }


def test_qlib_backend_requires_frequency_when_runner_returns_multiple() -> None:
    report, positions = _qlib_output()[0]["1day"]
    output = {"1day": (report, positions), "week": (report, positions)}
    request = QlibBacktestRequest("2020-01-01", "2020-01-03", "strategy", "executor")

    with pytest.raises(ValueError, match="multiple frequencies"):
        QlibBacktestBackend(runner=lambda **_: (output, {})).run(request)


def test_qlib_backend_preserves_intraday_timestamps() -> None:
    report, positions = _qlib_output()[0]["1day"]
    report.index = pd.to_datetime(["2020-01-02 09:35", "2020-01-02 09:40"])
    positions = {
        pd.Timestamp("2020-01-02 09:35"): {"AAA": 0.5},
        pd.Timestamp("2020-01-02 09:40"): {"AAA": 1.0},
    }
    request = QlibBacktestRequest(
        "2020-01-02 09:35",
        "2020-01-02 09:40",
        "strategy",
        "executor",
    )

    result = QlibBacktestBackend(runner=lambda **_: ({"5min": (report, positions)}, {})).run(
        request
    )

    assert result.performance["date"].tolist() == list(report.index)


def test_qlib_request_rejects_runtime_framework_objects() -> None:
    with pytest.raises(TypeError, match="strategy"):
        QlibBacktestRequest(  # type: ignore[arg-type]
            "2020-01-01",
            "2020-01-03",
            object(),
            "executor",
        )

    with pytest.raises(TypeError, match="JSON-compatible"):
        QlibBacktestRequest(
            "2020-01-01",
            "2020-01-03",
            {"kwargs": {"runtime_model": object()}},
            "executor",
        )


def test_qlib_backend_passes_defensive_config_copies_to_mutating_runner() -> None:
    request = QlibBacktestRequest(
        "2020-01-01",
        "2020-01-03",
        {"class": "Strategy", "kwargs": {"signals": [1, 2]}},
        {"class": "Executor", "kwargs": {}},
        account={"cash": 1_000_000.0},
        exchange_kwargs={"codes": ["AAA"]},
    )

    def mutating_runner(**kwargs: Any) -> object:
        kwargs["strategy"]["kwargs"]["signals"].append(3)
        kwargs["account"].pop("cash")
        kwargs["exchange_kwargs"]["codes"].append("BBB")
        return _qlib_output()

    QlibBacktestBackend(runner=mutating_runner).run(request)

    assert request.strategy == {"class": "Strategy", "kwargs": {"signals": [1, 2]}}
    assert request.account == {"cash": 1_000_000.0}
    assert request.exchange_kwargs == {"codes": ["AAA"]}


def test_qlib_backend_reports_missing_optional_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(_: str) -> object:
        error = ModuleNotFoundError("No module named 'qlib'")
        error.name = "qlib"
        raise error

    monkeypatch.setattr(qlib_adapter, "import_module", missing)

    with pytest.raises(QlibUnavailableError, match="optional 'qlib' extra"):
        qlib_adapter._load_official_qlib_backtest()


def test_loader_resolves_official_qlib_backtest_when_extra_is_installed() -> None:
    module = pytest.importorskip("qlib.backtest")

    assert qlib_adapter._load_official_qlib_backtest() is module.backtest
