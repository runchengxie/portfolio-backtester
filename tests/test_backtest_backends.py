from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import portfolio_backtester.backends.native as native_backend_module
from portfolio_backtester.backends import (
    BackendCapabilities,
    BackendRegistry,
    CanonicalBacktestResult,
    NativePositionReplayBackend,
    NativePositionReplayRequest,
)
from portfolio_backtester.position_backtest import PositionBacktestConfig

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "backends" / "liquid_long_only.expected.json"


def _request(**overrides) -> NativePositionReplayRequest:
    positions = pd.DataFrame(
        [
            {
                "rebalance_date": "20200101",
                "entry_date": "20200102",
                "symbol": "AAA",
                "weight": 0.75,
                "side": "long",
            },
            {
                "rebalance_date": "20200101",
                "entry_date": "20200102",
                "symbol": "BBB",
                "weight": 0.25,
                "side": "long",
            },
        ]
    )
    pricing = pd.DataFrame(
        [
            {"trade_date": "20200102", "symbol": "AAA", "close": 10.0},
            {"trade_date": "20200102", "symbol": "BBB", "close": 20.0},
            {"trade_date": "20200103", "symbol": "AAA", "close": 11.0},
            {"trade_date": "20200103", "symbol": "BBB", "close": 18.0},
        ]
    )
    periods = pd.DataFrame(
        [
            {
                "rebalance_date": "20200101",
                "entry_date": "20200102",
                "exit_date": "20200103",
            }
        ]
    )
    values = {
        "positions": positions,
        "pricing": pricing,
        "periods": periods,
        "config": PositionBacktestConfig(transaction_cost_bps=10.0),
    }
    values.update(overrides)
    return NativePositionReplayRequest(**values)


def test_native_backend_matches_committed_liquid_golden_result() -> None:
    expected = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    result = NativePositionReplayBackend().run(_request())

    assert result.describe()["schema_version"] == expected["schema_version"]
    assert result.backend_name == expected["backend_name"]
    records = result.performance.to_dict("records")
    assert len(records) == 1
    assert records[0]["period_end"] == expected["performance"][0]["period_end"]
    assert records[0]["net_return"] == pytest.approx(expected["performance"][0]["net_return"])
    assert records[0]["gross_return"] == pytest.approx(
        expected["performance"][0]["gross_return"]
    )
    assert result.capabilities.to_mapping() == expected["capabilities"]
    assert result.orders.empty
    assert result.fills.empty
    assert result.daily_ledger.empty


def test_native_backend_preserves_duplicate_period_end_rows(monkeypatch) -> None:
    class FakeResult:
        net_returns = pd.DataFrame(
            [
                {"period_end": "2020-01-03", "net_return": 0.01},
                {"period_end": "2020-01-03", "net_return": 0.02},
            ]
        )
        gross_returns = pd.DataFrame(
            [
                {"period_end": "2020-01-03", "gross_return": 0.011},
                {"period_end": "2020-01-03", "gross_return": 0.021},
            ]
        )
        summary = {"schema": "position_backtest.v1"}

    monkeypatch.setattr(
        native_backend_module,
        "run_position_backtest",
        lambda **_: FakeResult(),
    )

    result = NativePositionReplayBackend().run(_request())

    assert result.performance.shape[0] == 2
    assert result.performance["gross_return"].tolist() == pytest.approx([0.011, 0.021])


def test_native_backend_fails_closed_for_unsupported_short_positions() -> None:
    positions = _request().positions.copy()
    positions.loc[0, "side"] = "short"

    with pytest.raises(ValueError, match="does not support position side"):
        NativePositionReplayBackend().run(_request(positions=positions))


def test_native_backend_fails_closed_for_long_only_false() -> None:
    config = PositionBacktestConfig(long_only=False)

    with pytest.raises(ValueError, match="long_only=False"):
        NativePositionReplayBackend().run(_request(config=config))


def test_native_backend_requires_explicit_intraday_timing_assumption() -> None:
    intraday = pd.DataFrame(
        [
            {
                "trade_date": "20200102",
                "symbol": "AAA",
                "close": 10.0,
                "volume": 100,
            }
        ]
    )

    with pytest.raises(ValueError, match="intraday_execution_assumption"):
        NativePositionReplayBackend().run(_request(intraday_bars=intraday))


def test_native_backend_requires_explicit_stale_execution_opt_in() -> None:
    config = PositionBacktestConfig(exit_price_policy="ffill")

    with pytest.raises(ValueError, match="allow_stale_execution_price"):
        NativePositionReplayBackend().run(_request(config=config))


def test_backend_registry_rejects_duplicate_names() -> None:
    registry = BackendRegistry()
    backend = NativePositionReplayBackend()
    registry.register(backend)

    assert registry.names() == ("native.position_replay",)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(backend)


def test_canonical_result_rejects_fake_orders_from_non_order_backend() -> None:
    result = CanonicalBacktestResult(
        backend_name="period-only",
        performance=pd.DataFrame([{"period_end": "2020-01-03", "net_return": 0.01}]),
        positions=pd.DataFrame(
            [{"rebalance_date": "2020-01-01", "symbol": "AAA", "weight": 1.0}]
        ),
        capabilities=BackendCapabilities(order_lifecycle=False),
        orders=pd.DataFrame([{"order_id": "order-1", "status": "filled"}]),
    )

    with pytest.raises(ValueError, match="must not emit orders"):
        result.validate()
