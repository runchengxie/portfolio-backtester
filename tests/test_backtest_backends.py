from __future__ import annotations

import math

import pandas as pd
import pytest

from cstree.backtesting.backends import (
    BacktestBackendResult,
    NativeAShareReplayBackend,
    PositionReplayRequest,
)


def test_native_a_share_backend_preserves_delayed_exit_and_fee_semantics(
    a_share_replay_request: PositionReplayRequest,
) -> None:
    result = NativeAShareReplayBackend().run(a_share_replay_request)

    assert result.backend == "native-a-share-replay"
    assert result.metadata["canonical"] is True
    assert result.metadata["position_semantics"] == "target_rebalance"
    assert result.performance["date"].tolist() == [pd.Timestamp("2020-01-04")]
    assert result.positions.to_dict("records") == [
        {"date": pd.Timestamp("2020-01-01"), "symbol": "AAA", "weight": 0.5},
        {"date": pd.Timestamp("2020-01-01"), "symbol": "BBB", "weight": 0.5},
    ]
    period = result.performance.iloc[0]
    assert math.isclose(period["gross_return"], 0.05, abs_tol=1e-12)
    assert math.isclose(period["fee_cost"], 0.001, abs_tol=1e-12)
    assert math.isclose(period["net_return"], 0.049, abs_tol=1e-12)
    assert math.isclose(period["pnl"], 0.049, abs_tol=1e-12)


def test_backend_result_rejects_duplicate_dates_and_non_finite_values() -> None:
    performance = pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-01"],
            "gross_return": [0.0, 0.0],
            "net_return": [0.0, 0.0],
            "turnover": [0.0, 0.0],
            "fee_cost": [0.0, 0.0],
            "slippage_cost": [0.0, 0.0],
            "total_cost": [0.0, 0.0],
            "pnl": [0.0, float("nan")],
        }
    )
    positions = pd.DataFrame(columns=["date", "symbol", "weight"])

    with pytest.raises(ValueError, match="non-finite"):
        BacktestBackendResult("test", performance, positions, {})


def test_backend_result_rejects_runtime_objects_in_metadata() -> None:
    performance = pd.DataFrame(
        {
            "date": ["2020-01-01"],
            "gross_return": [0.0],
            "net_return": [0.0],
            "turnover": [0.0],
            "fee_cost": [0.0],
            "slippage_cost": [0.0],
            "total_cost": [0.0],
            "pnl": [0.0],
        }
    )
    positions = pd.DataFrame(columns=["date", "symbol", "weight"])

    with pytest.raises(TypeError, match="non-neutral value"):
        BacktestBackendResult("test", performance, positions, {"runtime": object()})


def test_position_replay_request_satisfies_backend_protocol(
    a_share_replay_request: PositionReplayRequest,
) -> None:
    backend = NativeAShareReplayBackend()

    result = backend.run(a_share_replay_request)

    assert result.performance.columns.tolist() == [
        "date",
        "gross_return",
        "net_return",
        "turnover",
        "fee_cost",
        "slippage_cost",
        "total_cost",
        "pnl",
    ]
