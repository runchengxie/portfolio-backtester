from __future__ import annotations

import math

import pandas as pd

from portfolio_backtester.position_backtest import (
    PositionBacktestConfig,
    run_position_backtest,
)


def test_position_backtest_uses_explicit_unequal_weights_and_turnover_costs() -> None:
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
            {
                "rebalance_date": "20200104",
                "entry_date": "20200105",
                "symbol": "AAA",
                "weight": 0.25,
                "side": "long",
            },
            {
                "rebalance_date": "20200104",
                "entry_date": "20200105",
                "symbol": "CCC",
                "weight": 0.75,
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
            {"trade_date": "20200105", "symbol": "AAA", "close": 12.0},
            {"trade_date": "20200105", "symbol": "BBB", "close": 19.0},
            {"trade_date": "20200105", "symbol": "CCC", "close": 30.0},
            {"trade_date": "20200106", "symbol": "AAA", "close": 12.6},
            {"trade_date": "20200106", "symbol": "CCC", "close": 33.0},
        ]
    )
    periods = pd.DataFrame(
        [
            {
                "rebalance_date": "20200101",
                "entry_idx": 0,
                "planned_exit_idx": 1,
                "exit_idx": 1,
                "entry_date": "20200102",
                "planned_exit_date": "20200103",
                "exit_date": "20200103",
                "exit_delay_steps": 0,
            },
            {
                "rebalance_date": "20200104",
                "entry_idx": 2,
                "planned_exit_idx": 3,
                "exit_idx": 3,
                "entry_date": "20200105",
                "planned_exit_date": "20200106",
                "exit_date": "20200106",
                "exit_delay_steps": 0,
            },
        ]
    )

    result = run_position_backtest(
        positions=positions,
        pricing=pricing,
        periods=periods,
        config=PositionBacktestConfig(transaction_cost_bps=10.0),
    )

    period_rows = result.periods
    assert period_rows["position_count"].tolist() == [2, 2]
    assert math.isclose(period_rows.loc[0, "gross_return"], 0.05, abs_tol=1e-12)
    assert math.isclose(period_rows.loc[0, "net_return"], 0.049, abs_tol=1e-12)

    drift_aaa = 0.75 * (12.0 / 10.0)
    drift_bbb = 0.25 * (19.0 / 20.0)
    drift_total = drift_aaa + drift_bbb
    drift_aaa_weight = drift_aaa / drift_total
    drift_bbb_weight = drift_bbb / drift_total
    expected_turnover = 0.5 * (
        abs(0.25 - drift_aaa_weight) + abs(0.0 - drift_bbb_weight) + abs(0.75 - 0.0)
    )
    expected_cost = 2.0 * 10.0 / 10000.0 * expected_turnover
    assert math.isclose(period_rows.loc[1, "turnover"], expected_turnover, abs_tol=1e-12)
    assert math.isclose(period_rows.loc[1, "fee_cost"], expected_cost, abs_tol=1e-12)
    assert result.summary["stats"]["weighting"] == "positions"
    assert result.net_returns.columns.tolist() == ["period_end", "net_return"]


def test_position_backtest_drops_missing_price_symbols_and_renormalizes() -> None:
    positions = pd.DataFrame(
        [
            {"rebalance_date": "20200101", "symbol": "AAA", "weight": 0.50},
            {"rebalance_date": "20200101", "symbol": "MISSING", "weight": 0.50},
        ]
    )
    pricing = pd.DataFrame(
        [
            {"trade_date": "20200102", "symbol": "AAA", "close": 10.0},
            {"trade_date": "20200103", "symbol": "AAA", "close": 11.0},
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

    result = run_position_backtest(
        positions=positions,
        pricing=pricing,
        periods=periods,
        config=PositionBacktestConfig(),
    )

    assert result.periods.loc[0, "missing_price_count"] == 1
    assert math.isclose(result.periods.loc[0, "gross_return"], 0.10, abs_tol=1e-12)


def test_position_backtest_preserves_cash_when_requested() -> None:
    positions = pd.DataFrame(
        [
            {"rebalance_date": "20200101", "symbol": "AAA", "weight": 0.25},
            {"rebalance_date": "20200101", "symbol": "BBB", "weight": 0.25},
        ]
    )
    pricing = pd.DataFrame(
        [
            {"trade_date": "20200102", "symbol": "AAA", "close": 10.0},
            {"trade_date": "20200102", "symbol": "BBB", "close": 20.0},
            {"trade_date": "20200103", "symbol": "AAA", "close": 11.0},
            {"trade_date": "20200103", "symbol": "BBB", "close": 22.0},
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

    default_result = run_position_backtest(
        positions=positions,
        pricing=pricing,
        periods=periods,
        config=PositionBacktestConfig(),
    )
    cash_result = run_position_backtest(
        positions=positions,
        pricing=pricing,
        periods=periods,
        config=PositionBacktestConfig(
            transaction_cost_bps=30.0,
            preserve_gross_exposure=True,
        ),
    )

    assert math.isclose(default_result.periods.loc[0, "gross_return"], 0.10, abs_tol=1e-12)
    assert math.isclose(cash_result.periods.loc[0, "gross_return"], 0.05, abs_tol=1e-12)
    assert math.isclose(cash_result.periods.loc[0, "gross_exposure"], 0.50, abs_tol=1e-12)
    assert math.isclose(cash_result.periods.loc[0, "cash_weight"], 0.50, abs_tol=1e-12)
    assert math.isclose(cash_result.periods.loc[0, "fee_cost"], 0.0015, abs_tol=1e-12)
    assert math.isclose(cash_result.periods.loc[0, "net_return"], 0.0485, abs_tol=1e-12)
    assert cash_result.summary["config"]["preserve_gross_exposure"] is True
    assert math.isclose(cash_result.summary["stats"]["avg_cash_weight"], 0.50, abs_tol=1e-12)


def test_position_backtest_delay_exit_resolves_prices_per_symbol() -> None:
    positions = pd.DataFrame(
        [
            {"rebalance_date": "20200101", "symbol": "AAA", "weight": 0.50},
            {"rebalance_date": "20200101", "symbol": "BBB", "weight": 0.50},
        ]
    )
    pricing = pd.DataFrame(
        [
            {"trade_date": "20200102", "symbol": "AAA", "close": 10.0, "is_tradable": True},
            {"trade_date": "20200102", "symbol": "BBB", "close": 20.0, "is_tradable": True},
            {"trade_date": "20200103", "symbol": "AAA", "close": 11.0, "is_tradable": False},
            {"trade_date": "20200103", "symbol": "BBB", "close": 18.0, "is_tradable": True},
            {"trade_date": "20200104", "symbol": "AAA", "close": 12.0, "is_tradable": True},
            {"trade_date": "20200104", "symbol": "BBB", "close": 22.0, "is_tradable": True},
        ]
    )
    periods = pd.DataFrame(
        [
            {
                "rebalance_date": "20200101",
                "entry_idx": 0,
                "planned_exit_idx": 0,
                "exit_idx": 2,
                "entry_date": "20200102",
                "planned_exit_date": 20200103,
                "exit_date": "20200104",
                "exit_delay_steps": 1,
            }
        ]
    )

    result = run_position_backtest(
        positions=positions,
        pricing=pricing,
        periods=periods,
        config=PositionBacktestConfig(
            exit_price_policy="delay",
            tradable_col="is_tradable",
        ),
    )

    period = result.periods.iloc[0]
    expected = 0.5 * ((12.0 / 10.0) - 1.0) + 0.5 * ((18.0 / 20.0) - 1.0)
    assert period["exit_date"] == "20200104"
    assert period["exit_idx"] == 2
    assert period["exit_delay_steps"] == 1
    assert math.isclose(period["gross_return"], expected, abs_tol=1e-12)
    assert result.summary["config"]["exit_price_policy"] == "delay"
