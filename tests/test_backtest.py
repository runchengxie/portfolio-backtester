import numpy as np
import pandas as pd
import pytest

from portfolio_backtester.engine import backtest_topk
from portfolio_backtester.execution import build_execution_model


def test_backtest_initial_cost_applied():
    df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02"]),
            "symbol": ["A", "B", "A", "B"],
            "pred": [2.0, 1.0, 2.0, 1.0],
            "close": [100.0, 100.0, 110.0, 90.0],
        }
    )
    rebalance_dates = [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")]
    result = backtest_topk(
        df,
        pred_col="pred",
        price_col="close",
        rebalance_dates=rebalance_dates,
        top_k=1,
        shift_days=0,
        cost_bps=10,
        trading_days_per_year=252,
        exit_mode="rebalance",
    )
    stats, net_series, gross_series, turnover_series, _ = result
    assert stats["periods"] == 1
    assert np.isclose(gross_series.iloc[0], 0.10)
    assert np.isclose(net_series.iloc[0], 0.10 - 0.001)
    assert np.isclose(turnover_series.iloc[0], 1.0)
    assert stats["periods_with_delayed_exit"] == 0


def test_backtest_rank_offset_skips_top_ranked_names():
    df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(
                [
                    "2020-01-01",
                    "2020-01-01",
                    "2020-01-01",
                    "2020-01-02",
                    "2020-01-02",
                    "2020-01-02",
                ]
            ),
            "symbol": ["A", "B", "C", "A", "B", "C"],
            "pred": [3.0, 2.0, 1.0, 3.0, 2.0, 1.0],
            "close": [100.0, 100.0, 100.0, 200.0, 150.0, 80.0],
        }
    )
    rebalance_dates = [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")]

    _, _, gross_series, _, _ = backtest_topk(
        df,
        pred_col="pred",
        price_col="close",
        rebalance_dates=rebalance_dates,
        top_k=1,
        shift_days=0,
        cost_bps=0,
        trading_days_per_year=252,
        exit_mode="rebalance",
        rank_offset=1,
    )

    assert np.isclose(gross_series.iloc[0], 0.50)


def test_backtest_accepts_legacy_ts_code_input():
    df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02"]),
            "ts_code": ["A", "B", "A", "B"],
            "pred": [2.0, 1.0, 2.0, 1.0],
            "close": [100.0, 100.0, 110.0, 90.0],
        }
    )
    rebalance_dates = [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")]

    stats, net_series, gross_series, turnover_series, _ = backtest_topk(
        df,
        pred_col="pred",
        price_col="close",
        rebalance_dates=rebalance_dates,
        top_k=1,
        shift_days=0,
        cost_bps=10,
        trading_days_per_year=252,
        exit_mode="rebalance",
    )

    assert stats["periods"] == 1
    assert np.isclose(gross_series.iloc[0], 0.10)
    assert np.isclose(net_series.iloc[0], 0.10 - 0.001)
    assert np.isclose(turnover_series.iloc[0], 1.0)


def test_backtest_turnover_accounts_for_weight_drift():
    df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(
                [
                    "2020-01-01",
                    "2020-01-01",
                    "2020-01-02",
                    "2020-01-02",
                    "2020-01-03",
                    "2020-01-03",
                ]
            ),
            "symbol": ["A", "B"] * 3,
            "pred": [2.0, 1.0] * 3,
            "close": [100.0, 100.0, 200.0, 100.0, 200.0, 100.0],
        }
    )
    rebalance_dates = [
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-01-02"),
        pd.Timestamp("2020-01-03"),
    ]
    result = backtest_topk(
        df,
        pred_col="pred",
        price_col="close",
        rebalance_dates=rebalance_dates,
        top_k=2,
        shift_days=0,
        cost_bps=0,
        trading_days_per_year=252,
        exit_mode="rebalance",
    )
    assert result is not None
    _, _, _, turnover_series, periods = result
    # First period is initial entry (turnover=1). Second period should reflect drift.
    assert turnover_series.shape[0] == 2
    assert np.isclose(turnover_series.iloc[1], 1 / 6, atol=1e-6)
    assert periods[1]["target_weight_full_l1"] == pytest.approx(0.0)
    assert periods[1]["target_weight_half_l1"] == pytest.approx(0.0)
    assert periods[1]["pretrade_demand_buy"] == pytest.approx(1 / 6)
    assert periods[1]["pretrade_demand_sell"] == pytest.approx(1 / 6)
    assert periods[1]["pretrade_demand_full_l1"] == pytest.approx(1 / 3)
    assert periods[1]["pretrade_demand_half_l1"] == pytest.approx(1 / 6)


def test_backtest_label_horizon_overlap_raises():
    df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(
                [
                    "2020-01-01",
                    "2020-01-01",
                    "2020-01-02",
                    "2020-01-02",
                    "2020-01-03",
                    "2020-01-03",
                    "2020-01-04",
                    "2020-01-04",
                ]
            ),
            "symbol": ["A", "B"] * 4,
            "pred": [2.0, 1.0] * 4,
            "close": [100.0, 100.0, 101.0, 99.0, 102.0, 98.0, 103.0, 97.0],
        }
    )
    rebalance_dates = [
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-01-02"),
        pd.Timestamp("2020-01-03"),
    ]
    with pytest.raises(ValueError):
        backtest_topk(
            df,
            pred_col="pred",
            price_col="close",
            rebalance_dates=rebalance_dates,
            top_k=1,
            shift_days=0,
            cost_bps=0,
            trading_days_per_year=252,
            exit_mode="label_horizon",
            exit_horizon_days=2,
        )


def test_backtest_long_short_basic():
    df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02"]),
            "symbol": ["A", "B", "A", "B"],
            "pred": [2.0, 1.0, 2.0, 1.0],
            "close": [100.0, 100.0, 110.0, 90.0],
        }
    )
    rebalance_dates = [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")]
    result = backtest_topk(
        df,
        pred_col="pred",
        price_col="close",
        rebalance_dates=rebalance_dates,
        top_k=1,
        shift_days=0,
        cost_bps=0,
        trading_days_per_year=252,
        exit_mode="rebalance",
        long_only=False,
        short_k=1,
    )
    stats, net_series, gross_series, turnover_series, _ = result
    assert stats["periods"] == 1
    assert np.isclose(gross_series.iloc[0], 0.2)
    assert np.isclose(net_series.iloc[0], 0.2)
    assert np.isclose(turnover_series.iloc[0], 2.0)


def test_backtest_signal_weighting_uses_signal_magnitude():
    df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02"]),
            "symbol": ["A", "B", "A", "B"],
            "pred": [3.0, 1.0, 3.0, 1.0],
            "close": [100.0, 100.0, 120.0, 100.0],
        }
    )
    rebalance_dates = [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")]
    result = backtest_topk(
        df,
        pred_col="pred",
        price_col="close",
        rebalance_dates=rebalance_dates,
        top_k=2,
        shift_days=0,
        cost_bps=0,
        trading_days_per_year=252,
        exit_mode="rebalance",
        weighting="signal",
    )
    stats, net_series, gross_series, turnover_series, _ = result
    raw = np.exp(np.array([1.0, -1.0]))
    expected_weights = raw / raw.sum()
    expected_gross = expected_weights[0] * 0.20 + expected_weights[1] * 0.0
    assert stats["weighting"] == "signal"
    assert np.isclose(gross_series.iloc[0], expected_gross)
    assert np.isclose(net_series.iloc[0], expected_gross)
    assert np.isclose(turnover_series.iloc[0], 1.0)


def test_backtest_group_cap_limits_names_per_group():
    df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(
                [
                    "2020-01-01",
                    "2020-01-01",
                    "2020-01-01",
                    "2020-01-01",
                    "2020-01-01",
                    "2020-01-01",
                    "2020-01-02",
                    "2020-01-02",
                    "2020-01-02",
                    "2020-01-02",
                    "2020-01-02",
                    "2020-01-02",
                ]
            ),
            "symbol": ["A1", "A2", "B1", "B2", "C1", "C2"] * 2,
            "pred": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0] * 2,
            "close": [
                100.0,
                100.0,
                100.0,
                100.0,
                100.0,
                100.0,
                120.0,
                80.0,
                110.0,
                100.0,
                130.0,
                100.0,
            ],
            "industry": ["A", "A", "B", "B", "C", "C"] * 2,
        }
    )
    rebalance_dates = [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")]
    result = backtest_topk(
        df,
        pred_col="pred",
        price_col="close",
        rebalance_dates=rebalance_dates,
        top_k=3,
        shift_days=0,
        cost_bps=0,
        trading_days_per_year=252,
        exit_mode="rebalance",
        group_col="industry",
        max_names_per_group=1,
    )
    stats, net_series, gross_series, _, _period_info = result
    expected_gross = np.mean([0.20, 0.10, 0.30])
    assert stats["periods"] == 1
    assert np.isclose(gross_series.iloc[0], expected_gross)
    assert np.isclose(net_series.iloc[0], expected_gross)


def test_backtest_exit_delay_uses_next_available_price():
    df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(
                ["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-03", "2020-01-03"]
            ),
            "symbol": ["A", "B", "B", "A", "B"],
            "pred": [2.0, 1.0, 1.0, 2.0, 1.0],
            "close": [100.0, 100.0, 100.0, 90.0, 100.0],
        }
    )
    rebalance_dates = [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")]
    result = backtest_topk(
        df,
        pred_col="pred",
        price_col="close",
        rebalance_dates=rebalance_dates,
        top_k=1,
        shift_days=0,
        cost_bps=0,
        trading_days_per_year=252,
        exit_mode="rebalance",
        exit_price_policy="delay",
    )
    stats, net_series, _, _, period_info = result
    assert net_series.index[0] == pd.Timestamp("2020-01-03")
    assert np.isclose(net_series.iloc[0], -0.10)
    assert stats["periods_with_delayed_exit"] == 1
    assert np.isclose(stats["avg_exit_lag_days"], 1.0)
    assert np.isclose(stats["max_exit_lag_days"], 1.0)
    assert period_info[0]["planned_exit_date"] == pd.Timestamp("2020-01-02")
    assert period_info[0]["exit_date"] == pd.Timestamp("2020-01-03")
    assert period_info[0]["exit_delay_steps"] == 1


def test_backtest_can_exit_with_raw_pricing_data_after_selection_filter():
    filtered_df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02"]),
            "symbol": ["A", "B", "B"],
            "pred": [2.0, 1.0, 1.0],
            "close": [100.0, 100.0, 100.0],
        }
    )
    pricing_df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02"]),
            "symbol": ["A", "B", "A", "B"],
            "close": [100.0, 100.0, 110.0, 100.0],
        }
    )
    rebalance_dates = [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")]

    assert (
        backtest_topk(
            filtered_df,
            pred_col="pred",
            price_col="close",
            rebalance_dates=rebalance_dates,
            top_k=1,
            shift_days=0,
            cost_bps=0,
            trading_days_per_year=252,
            exit_mode="rebalance",
            exit_price_policy="strict",
        )
        is None
    )

    result = backtest_topk(
        filtered_df,
        pred_col="pred",
        price_col="close",
        rebalance_dates=rebalance_dates,
        top_k=1,
        shift_days=0,
        cost_bps=0,
        trading_days_per_year=252,
        exit_mode="rebalance",
        exit_price_policy="strict",
        pricing_data=pricing_df,
    )
    assert result is not None
    stats, net_series, _, _, period_info = result
    assert np.isclose(net_series.iloc[0], 0.10)
    assert stats["periods_with_delayed_exit"] == 0
    assert period_info[0]["planned_exit_date"] == pd.Timestamp("2020-01-02")
    assert period_info[0]["exit_date"] == pd.Timestamp("2020-01-02")


def test_backtest_buffer_reduces_turnover():
    df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(
                [
                    "2020-01-01",
                    "2020-01-01",
                    "2020-01-02",
                    "2020-01-02",
                    "2020-01-03",
                    "2020-01-03",
                ]
            ),
            "symbol": ["A", "B"] * 3,
            "pred": [2.0, 1.0, 1.0, 2.0, 1.0, 2.0],
            "close": [100.0] * 6,
        }
    )
    rebalance_dates = [
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-01-02"),
        pd.Timestamp("2020-01-03"),
    ]
    result = backtest_topk(
        df,
        pred_col="pred",
        price_col="close",
        rebalance_dates=rebalance_dates,
        top_k=1,
        shift_days=0,
        cost_bps=0,
        trading_days_per_year=252,
        exit_mode="rebalance",
        buffer_exit=1,
        buffer_entry=0,
    )
    _, _, _, turnover_series, _ = result
    assert turnover_series.shape[0] == 2
    assert np.isclose(turnover_series.iloc[1], 0.0)


def test_backtest_exit_strict_skips_missing_price():
    df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "symbol": ["A", "A"],
            "pred": [1.0, 1.0],
            "close": [100.0, np.nan],
        }
    )
    rebalance_dates = [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")]
    result = backtest_topk(
        df,
        pred_col="pred",
        price_col="close",
        rebalance_dates=rebalance_dates,
        top_k=1,
        shift_days=0,
        cost_bps=0,
        trading_days_per_year=252,
        exit_mode="rebalance",
        exit_price_policy="strict",
    )
    assert result is None


def test_backtest_exit_ffill_uses_last_price():
    df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "symbol": ["A", "A"],
            "pred": [1.0, 1.0],
            "close": [100.0, np.nan],
        }
    )
    rebalance_dates = [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")]
    result = backtest_topk(
        df,
        pred_col="pred",
        price_col="close",
        rebalance_dates=rebalance_dates,
        top_k=1,
        shift_days=0,
        cost_bps=0,
        trading_days_per_year=252,
        exit_mode="rebalance",
        exit_price_policy="ffill",
    )
    assert result is not None
    _, net_series, _, _, _ = result
    assert net_series.index[0] == pd.Timestamp("2020-01-02")
    assert np.isclose(net_series.iloc[0], 0.0)


def test_backtest_tradable_filters_entry_selection():
    df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02"]),
            "symbol": ["A", "B", "A", "B"],
            "pred": [2.0, 1.0, 2.0, 1.0],
            "close": [100.0, 100.0, 110.0, 90.0],
            "is_tradable": [False, True, False, True],
        }
    )
    rebalance_dates = [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")]
    result = backtest_topk(
        df,
        pred_col="pred",
        price_col="close",
        rebalance_dates=rebalance_dates,
        top_k=1,
        shift_days=0,
        cost_bps=0,
        trading_days_per_year=252,
        exit_mode="rebalance",
        exit_price_policy="strict",
        tradable_col="is_tradable",
    )
    assert result is not None
    _, net_series, _, _, _ = result
    assert np.isclose(net_series.iloc[0], -0.10)


def test_backtest_requires_declared_tradable_column():
    df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02"]),
            "symbol": ["A", "B", "A", "B"],
            "pred": [2.0, 1.0, 2.0, 1.0],
            "close": [100.0, 100.0, 110.0, 90.0],
        }
    )
    rebalance_dates = [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")]

    with pytest.raises(ValueError, match="missing columns: is_tradable"):
        backtest_topk(
            df,
            pred_col="pred",
            price_col="close",
            rebalance_dates=rebalance_dates,
            top_k=1,
            shift_days=0,
            cost_bps=0,
            trading_days_per_year=252,
            exit_mode="rebalance",
            tradable_col="is_tradable",
        )


def test_backtest_exit_delay_with_none_fallback_skips_unresolved_exit():
    df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
            "symbol": ["A", "A", "A"],
            "pred": [1.0, 1.0, 1.0],
            "close": [100.0, np.nan, np.nan],
            "is_tradable": [True, False, False],
        }
    )
    rebalance_dates = [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")]
    result = backtest_topk(
        df,
        pred_col="pred",
        price_col="close",
        rebalance_dates=rebalance_dates,
        top_k=1,
        shift_days=0,
        cost_bps=0,
        trading_days_per_year=252,
        exit_mode="rebalance",
        exit_price_policy="delay",
        exit_fallback_policy="none",
        tradable_col="is_tradable",
    )
    assert result is None


def test_backtest_exit_delay_with_ffill_fallback_uses_previous_tradable_price():
    df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
            "symbol": ["A", "A", "A"],
            "pred": [1.0, 1.0, 1.0],
            "close": [100.0, 99.0, 98.0],
            "is_tradable": [True, False, False],
        }
    )
    rebalance_dates = [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")]
    result = backtest_topk(
        df,
        pred_col="pred",
        price_col="close",
        rebalance_dates=rebalance_dates,
        top_k=1,
        shift_days=0,
        cost_bps=0,
        trading_days_per_year=252,
        exit_mode="rebalance",
        exit_price_policy="delay",
        exit_fallback_policy="ffill",
        tradable_col="is_tradable",
    )
    assert result is not None
    _, net_series, _, _, _ = result
    assert net_series.index[0] == pd.Timestamp("2020-01-02")
    assert np.isclose(net_series.iloc[0], 0.0)


def test_backtest_execution_can_use_open_entry_price():
    df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02"]),
            "symbol": ["A", "B", "A", "B"],
            "pred": [2.0, 1.0, 2.0, 1.0],
            "open": [100.0, 100.0, 111.0, 100.0],
            "close": [110.0, 100.0, 120.0, 100.0],
        }
    )
    execution = build_execution_model(
        {"entry": {"price_col": "open"}},
        default_cost_bps=0.0,
        default_exit_price_policy="strict",
        default_exit_fallback_policy="ffill",
    )
    rebalance_dates = [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")]
    result = backtest_topk(
        df,
        pred_col="pred",
        price_col="close",
        rebalance_dates=rebalance_dates,
        top_k=1,
        shift_days=0,
        cost_bps=0,
        trading_days_per_year=252,
        exit_mode="rebalance",
        execution=execution,
        pricing_data=df,
    )
    assert result is not None
    _, net_series, _, _, _ = result
    assert np.isclose(net_series.iloc[0], 0.20)


def test_backtest_execution_min_amount_filters_illiquid_entries():
    df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02"]),
            "symbol": ["A", "B", "A", "B"],
            "pred": [2.0, 1.0, 2.0, 1.0],
            "close": [100.0, 100.0, 110.0, 90.0],
            "amount": [10.0, 1000.0, 10.0, 1000.0],
        }
    )
    execution = build_execution_model(
        {"constraints": {"min_amount": 100.0, "amount_col": "amount"}},
        default_cost_bps=0.0,
        default_exit_price_policy="strict",
        default_exit_fallback_policy="ffill",
    )
    rebalance_dates = [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")]
    result = backtest_topk(
        df,
        pred_col="pred",
        price_col="close",
        rebalance_dates=rebalance_dates,
        top_k=1,
        shift_days=0,
        cost_bps=0,
        trading_days_per_year=252,
        exit_mode="rebalance",
        execution=execution,
        pricing_data=df,
    )
    assert result is not None
    _, net_series, _, _, _ = result
    assert np.isclose(net_series.iloc[0], -0.10)


def test_backtest_side_cost_and_bps_slippage_are_applied():
    df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02"]),
            "symbol": ["A", "B", "A", "B"],
            "pred": [2.0, 1.0, 2.0, 1.0],
            "close": [100.0, 100.0, 100.0, 100.0],
        }
    )
    execution = build_execution_model(
        {
            "cost": {"name": "side_bps", "buy_bps": 10, "sell_bps": 0},
            "slippage": {"name": "bps", "bps": 5},
        },
        default_cost_bps=0.0,
        default_exit_price_policy="strict",
        default_exit_fallback_policy="ffill",
    )
    rebalance_dates = [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")]
    result = backtest_topk(
        df,
        pred_col="pred",
        price_col="close",
        rebalance_dates=rebalance_dates,
        top_k=1,
        shift_days=0,
        cost_bps=0,
        trading_days_per_year=252,
        exit_mode="rebalance",
        execution=execution,
        pricing_data=df,
    )
    assert result is not None
    stats, net_series, _, _, _ = result
    assert np.isclose(net_series.iloc[0], -0.0015)
    assert np.isclose(stats["avg_fee_drag"], 0.0010)
    assert np.isclose(stats["avg_slippage_drag"], 0.0005)


def test_backtest_participation_slippage_uses_amount_column():
    df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02"]),
            "symbol": ["A", "B", "A", "B"],
            "pred": [2.0, 1.0, 2.0, 1.0],
            "close": [100.0, 100.0, 100.0, 100.0],
            "amount": [10000.0, 10000.0, 10000.0, 10000.0],
        }
    )
    execution = build_execution_model(
        {
            "cost": "none",
            "slippage": {
                "name": "participation",
                "base_bps": 0.0,
                "impact_bps": 100.0,
                "portfolio_value": 1000.0,
                "amount_col": "amount",
                "power": 1.0,
            },
        },
        default_cost_bps=0.0,
        default_exit_price_policy="strict",
        default_exit_fallback_policy="ffill",
    )
    rebalance_dates = [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")]
    result = backtest_topk(
        df,
        pred_col="pred",
        price_col="close",
        rebalance_dates=rebalance_dates,
        top_k=1,
        shift_days=0,
        cost_bps=0,
        trading_days_per_year=252,
        exit_mode="rebalance",
        execution=execution,
        pricing_data=df,
    )
    assert result is not None
    stats, net_series, _, _, _ = result
    assert np.isclose(net_series.iloc[0], -0.0010)
    assert np.isclose(stats["avg_slippage_drag"], 0.0010)
