from __future__ import annotations

import pandas as pd
import pytest

from portfolio_backtester import (
    build_position_replay_periods,
    positions_by_rebalance_from_targets,
    run_native_position_replay,
)
from portfolio_backtester.execution import ParticipationSlippageModel
from portfolio_backtester.execution_sim import ExecutionSimConfig
from portfolio_backtester.position_backtest import PositionBacktestConfig


def test_build_position_replay_periods_closes_each_target_at_next_entry() -> None:
    positions = positions_by_rebalance_from_targets(
        {
            pd.Timestamp("2024-01-02"): {"AAA": 1.0},
            pd.Timestamp("2024-02-01"): {"AAA": 1.0},
        },
        entry_dates={
            pd.Timestamp("2024-01-02"): pd.Timestamp("2024-01-03"),
            pd.Timestamp("2024-02-01"): pd.Timestamp("2024-02-02"),
        },
    )
    pricing = pd.DataFrame(
        [
            {"trade_date": "20240103", "symbol": "AAA", "close": 10.0},
            {"trade_date": "20240202", "symbol": "AAA", "close": 11.0},
            {"trade_date": "20240205", "symbol": "AAA", "close": 12.0},
        ]
    )

    periods = build_position_replay_periods(positions, pricing)

    assert periods.to_dict("records") == [
        {
            "rebalance_date": pd.Timestamp("2024-01-02"),
            "entry_date": pd.Timestamp("2024-01-03"),
            "exit_date": pd.Timestamp("2024-02-02"),
        },
        {
            "rebalance_date": pd.Timestamp("2024-02-01"),
            "entry_date": pd.Timestamp("2024-02-02"),
            "exit_date": pd.Timestamp("2024-02-05"),
        },
    ]


def test_run_native_position_replay_returns_canonical_performance() -> None:
    positions = positions_by_rebalance_from_targets(
        {pd.Timestamp("2024-01-02"): {"AAA": 1.0}},
        entry_dates={pd.Timestamp("2024-01-02"): pd.Timestamp("2024-01-03")},
    )
    pricing = pd.DataFrame(
        [
            {"trade_date": "20240103", "symbol": "AAA", "close": 10.0},
            {"trade_date": "20240104", "symbol": "AAA", "close": 11.0},
        ]
    )
    periods = pd.DataFrame(
        [
            {
                "rebalance_date": "20240102",
                "entry_date": "20240103",
                "exit_date": "20240104",
            }
        ]
    )

    result = run_native_position_replay(
        positions,
        pricing,
        periods,
        config=PositionBacktestConfig(price_col="close"),
    )

    assert result.backend_name == "native.position_replay"
    assert result.performance.loc[0, "net_return"] == pytest.approx(0.1)
    assert result.orders.empty


def test_run_native_position_replay_forwards_ledger_and_slippage() -> None:
    positions = positions_by_rebalance_from_targets(
        {pd.Timestamp("2024-01-02"): {"AAA": 1.0}},
        entry_dates={pd.Timestamp("2024-01-02"): pd.Timestamp("2024-01-03")},
    )
    pricing = pd.DataFrame(
        [
            {
                "trade_date": "20240103",
                "symbol": "AAA",
                "close": 10.0,
                "amount": 10_000.0,
            },
            {
                "trade_date": "20240104",
                "symbol": "AAA",
                "close": 11.0,
                "amount": 10_000.0,
            },
        ]
    )
    periods = pd.DataFrame(
        [
            {
                "rebalance_date": "20240102",
                "entry_date": "20240103",
                "exit_date": "20240104",
            }
        ]
    )

    result = run_native_position_replay(
        positions,
        pricing,
        periods,
        config=PositionBacktestConfig(price_col="close", tradable_col="amount"),
        ledger=True,
        ledger_config=ExecutionSimConfig(
            enabled=True,
            portfolio_value=1_000.0,
            participation_rate=1.0,
            liquidity_cols=("amount",),
        ),
        slippage_model=ParticipationSlippageModel(
            impact_bps=100.0,
            amount_col="amount",
            portfolio_value=1_000.0,
        ),
    )

    assert result.capabilities.daily_ledger is True
    assert result.fills["cost_temporary_impact"].sum() > 0.0
