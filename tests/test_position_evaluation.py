from __future__ import annotations

import math

import pandas as pd
import pytest

from portfolio_backtester.position_backtest import PositionBacktestConfig
from portfolio_backtester.position_evaluation import evaluate_position_backtest


def _single_period_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    positions = pd.DataFrame([{"rebalance_date": "20200101", "symbol": "AAA", "weight": 1.0}])
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
    return positions, pricing, periods


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


def test_evaluate_position_backtest_normalizes_benchmark_price_frame() -> None:
    positions, pricing, periods = _single_period_inputs()
    benchmark_prices = pd.DataFrame(
        [
            {"trade_date": "20200102", "close": "200"},
            {"trade_date": "20200103", "close": "206"},
        ]
    )

    evaluation = evaluate_position_backtest(
        positions=positions,
        pricing=pricing,
        periods=periods,
        config=PositionBacktestConfig(),
        benchmark_df=benchmark_prices,
    )

    assert evaluation.benchmark_returns["benchmark_return"].tolist() == pytest.approx([0.03])
    assert evaluation.active_stats["n"] == 1


def test_evaluate_position_backtest_rejects_duplicate_benchmark_dates() -> None:
    positions, pricing, periods = _single_period_inputs()
    benchmark_prices = pd.DataFrame(
        [
            {"trade_date": "20200102", "close": 200.0},
            {"trade_date": "20200102", "close": 201.0},
            {"trade_date": "20200103", "close": 206.0},
        ]
    )

    with pytest.raises(ValueError, match="at most one row per trade_date"):
        evaluate_position_backtest(
            positions=positions,
            pricing=pricing,
            periods=periods,
            config=PositionBacktestConfig(),
            benchmark_df=benchmark_prices,
        )


def test_evaluate_position_backtest_preserves_duplicate_exit_dates() -> None:
    positions = pd.DataFrame(
        [
            {"rebalance_date": "20200101", "symbol": "AAA", "weight": 1.0},
            {"rebalance_date": "20200102", "symbol": "AAA", "weight": 1.0},
        ]
    )
    pricing = pd.DataFrame(
        [
            {"trade_date": "20200102", "symbol": "AAA", "close": 100.0},
            {"trade_date": "20200103", "symbol": "AAA", "close": 105.0},
            {"trade_date": "20200104", "symbol": "AAA", "close": 110.0},
        ]
    )
    periods = pd.DataFrame(
        [
            {
                "rebalance_date": "20200101",
                "entry_idx": 0,
                "exit_idx": 2,
                "entry_date": "20200102",
                "exit_date": "20200104",
            },
            {
                "rebalance_date": "20200102",
                "entry_idx": 1,
                "exit_idx": 2,
                "entry_date": "20200103",
                "exit_date": "20200104",
            },
        ]
    )
    benchmark = pd.Series(
        [0.01, 0.02],
        index=pd.to_datetime(["2020-01-03", "2020-01-04"]),
        name="benchmark_return",
    )

    evaluation = evaluate_position_backtest(
        positions=positions,
        pricing=pricing,
        periods=periods,
        config=PositionBacktestConfig(),
        benchmark_return_series=benchmark,
    )

    assert evaluation.benchmark_returns["benchmark_return"].tolist() == pytest.approx(
        [0.0302, 0.02]
    )
    assert evaluation.benchmark_returns["period_end"].nunique() == 1
    assert evaluation.active_stats["n"] == 2


def test_evaluate_position_backtest_returns_empty_active_summary_without_overlap() -> None:
    positions, pricing, periods = _single_period_inputs()
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
