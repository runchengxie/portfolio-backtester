from __future__ import annotations

import math

import pandas as pd

from portfolio_backtester.position_backtest import PositionBacktestConfig
from portfolio_backtester.position_evaluation import evaluate_position_backtest


def test_evaluate_position_backtest_compounds_daily_benchmark_returns() -> None:
    positions = pd.DataFrame(
        [
            {"rebalance_date": "20200101", "symbol": "AAA", "weight": 1.0},
            {"rebalance_date": "20200104", "symbol": "AAA", "weight": 1.0},
        ]
    )
    pricing = pd.DataFrame(
        [
            {"trade_date": "20200102", "symbol": "AAA", "close": 100.0},
            {"trade_date": "20200103", "symbol": "AAA", "close": 110.0},
            {"trade_date": "20200105", "symbol": "AAA", "close": 110.0},
            {"trade_date": "20200106", "symbol": "AAA", "close": 121.0},
        ]
    )
    periods = pd.DataFrame(
        [
            {
                "rebalance_date": "20200101",
                "entry_idx": 0,
                "exit_idx": 1,
                "entry_date": "20200102",
                "exit_date": "20200103",
            },
            {
                "rebalance_date": "20200104",
                "entry_idx": 2,
                "exit_idx": 3,
                "entry_date": "20200105",
                "exit_date": "20200106",
            },
        ]
    )
    benchmark = pd.Series(
        [0.02, 0.03, -0.01, 0.04],
        index=pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-05", "2020-01-06"]),
        name="benchmark_return",
    )

    evaluation = evaluate_position_backtest(
        positions=positions,
        pricing=pricing,
        periods=periods,
        config=PositionBacktestConfig(),
        benchmark_return_series=benchmark,
    )

    assert evaluation.benchmark_returns["benchmark_return"].tolist() == [0.03, 0.04]
    assert evaluation.active_stats["n"] == 2
    assert math.isfinite(evaluation.active_stats["tracking_error"])
    assert evaluation.summary["active_stats"] == evaluation.active_stats


def test_evaluate_position_backtest_returns_empty_active_summary_without_overlap() -> None:
    positions = pd.DataFrame(
        [{"rebalance_date": "20200101", "symbol": "AAA", "weight": 1.0}]
    )
    pricing = pd.DataFrame(
        [
            {"trade_date": "20200102", "symbol": "AAA", "close": 100.0},
            {"trade_date": "20200103", "symbol": "AAA", "close": 110.0},
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
    benchmark = pd.Series(
        [0.01],
        index=pd.to_datetime(["2021-01-01"]),
        name="benchmark_return",
    )

    evaluation = evaluate_position_backtest(
        positions=positions,
        pricing=pricing,
        periods=periods,
        config=PositionBacktestConfig(),
        benchmark_return_series=benchmark,
    )

    assert evaluation.active_stats["n"] == 0
    assert evaluation.active_returns.empty
