import pandas as pd
import pytest

from cstree.backtesting.execution import build_execution_model
from cstree.backtesting.position_postprocess import (
    apply_position_postprocess,
    rebuild_backtest_from_positions,
)


def test_cash_gross_overlay_scales_weights_from_schedule(tmp_path):
    schedule = tmp_path / "gross.csv"
    pd.DataFrame(
        {
            "rebalance_date": ["20200101", "20200108"],
            "target_gross": [0.90, 0.80],
        }
    ).to_csv(schedule, index=False)
    positions = pd.DataFrame(
        {
            "rebalance_date": [20200101, 20200101, 20200108, 20200108],
            "entry_date": [20200102, 20200102, 20200109, 20200109],
            "symbol": ["AAA", "BBB", "AAA", "BBB"],
            "weight": [0.6, 0.4, 0.5, 0.5],
            "signal": [2.0, 1.0, 2.0, 1.0],
            "rank": [1, 2, 1, 2],
            "side": ["long", "long", "long", "long"],
        }
    )

    overlaid, metadata, artifacts = apply_position_postprocess(
        positions,
        eval_df_full=pd.DataFrame(),
        context={
            "post_buffer_exposure_repair": {"enabled": False},
            "cash_gross_overlay": {"enabled": True, "schedule_file": str(schedule)},
        },
    )

    gross = overlaid.groupby("rebalance_date")["weight"].sum().to_dict()
    assert gross[20200101] == pytest.approx(0.90)
    assert gross[20200108] == pytest.approx(0.80)
    assert metadata["cash_gross_overlay"]["period_count"] == 2
    assert artifacts == {}
    assert "weight_before_cash_overlay" in overlaid.columns


def test_cash_gross_overlay_honors_top_level_tier_conditions():
    positions = pd.DataFrame(
        {
            "rebalance_date": [20200101, 20200101, 20200101, 20200108, 20200108],
            "entry_date": [20200102, 20200102, 20200102, 20200109, 20200109],
            "symbol": ["AAA", "BBB", "CCC", "AAA", "BBB"],
            "weight": [1 / 3, 1 / 3, 1 / 3, 0.5, 0.5],
            "signal": [3.0, 2.0, 1.0, 2.0, 1.0],
            "rank": [1, 2, 3, 1, 2],
            "side": ["long", "long", "long", "long", "long"],
        }
    )
    diagnostics = pd.DataFrame(
        {
            "trade_date": [20200101, 20200108],
            "momentum_active_net_vs_cap": [-1.2, -1.2],
        }
    )

    overlaid, metadata, _ = apply_position_postprocess(
        positions,
        eval_df_full=diagnostics,
        context={
            "post_buffer_exposure_repair": {"enabled": False},
            "cash_gross_overlay": {
                "enabled": True,
                "default_gross_multiplier": 1.0,
                "tiers": [
                    {
                        "min_position_count": 3,
                        "max_momentum_active_net_vs_cap": -1.0,
                        "gross_multiplier": 0.5,
                    }
                ],
            },
        },
    )

    gross = overlaid.groupby("rebalance_date")["weight"].sum().to_dict()
    assert gross[20200101] == pytest.approx(0.5)
    assert gross[20200108] == pytest.approx(1.0)
    assert metadata["cash_gross_overlay"]["min_target_gross"] == pytest.approx(0.5)
    assert metadata["cash_gross_overlay"]["max_target_gross"] == pytest.approx(1.0)


def test_auto_exposure_repair_generates_breaches_without_file():
    positions = pd.DataFrame(
        {
            "rebalance_date": [20200101, 20200101, 20200101],
            "entry_date": [20200102, 20200102, 20200102],
            "symbol": ["BANK", "GROWTH_A", "GROWTH_B"],
            "weight": [0.05, 0.50, 0.45],
            "signal": [0.5, 1.0, 0.8],
            "rank": [3, 1, 2],
            "side": ["long", "long", "long"],
        }
    )
    source = pd.DataFrame(
        {
            "trade_date": ["20200101", "20200101", "20200101", "20200101"],
            "symbol": ["BANK", "GROWTH_A", "GROWTH_B", "CASHLIKE"],
            "first_industry_name": ["银行", "电子", "传媒", "公用事业"],
            "signal_z": [0.5, 1.0, 0.8, 0.1],
            "earnings_burst_rank": [0.80, 0.75, 0.70, 0.60],
            "momentum": [-1.0, -1.0, -1.0, -1.0],
            "exposure_momentum_z": [-1.0, -1.0, -1.0, -1.0],
            "is_tradable": [True, True, True, True],
        }
    )

    repaired, metadata, artifacts = apply_position_postprocess(
        positions,
        eval_df_full=source,
        context={
            "post_buffer_exposure_repair": {
                "enabled": True,
                "max_abs_industry_active": 0.10,
                "max_abs_momentum_active": 1.0,
                "bank_industry_name": "银行",
            },
            "cash_gross_overlay": {"enabled": False},
            "price_col": "close",
            "backtest_pricing_df": pd.DataFrame(),
            "benchmark_df": None,
            "benchmark_return_series": pd.Series(dtype=float),
            "fundamentals_mcap_col": None,
            "industry_columns": ["first_industry_name"],
            "industry_source_df": pd.DataFrame(),
        },
    )

    repair_meta = metadata["post_buffer_exposure_repair"]
    bank_weight = repaired.loc[repaired["symbol"] == "BANK", "weight"].iloc[0]
    assert repair_meta["breach_source"] == "auto_exposure"
    assert repair_meta["breach_count"] == 1
    assert repair_meta["action_count"] == 1
    assert bank_weight > 0.05
    assert set(artifacts) == {
        "pre_repair_style",
        "pre_repair_industry",
        "pre_repair_active_summary",
        "breaches",
    }
    assert artifacts["breaches"].iloc[0]["check"] == "industry_active"


def test_rebuild_backtest_uses_postprocessed_positions_gross(tmp_path):
    positions = pd.DataFrame(
        {
            "rebalance_date": [20200101, 20200101],
            "entry_date": [20200102, 20200102],
            "symbol": ["AAA", "BBB"],
            "weight": [0.54, 0.36],
            "side": ["long", "long"],
        }
    )
    pricing = pd.DataFrame(
        {
            "trade_date": ["20200102", "20200102", "20200103", "20200103"],
            "symbol": ["AAA", "BBB", "AAA", "BBB"],
            "close": [10.0, 20.0, 11.0, 18.0],
            "is_tradable": [True, True, True, True],
        }
    )
    period_info = [
        {
            "rebalance_date": pd.Timestamp("2020-01-01"),
            "entry_date": pd.Timestamp("2020-01-02"),
            "planned_exit_date": pd.Timestamp("2020-01-03"),
            "exit_date": pd.Timestamp("2020-01-03"),
            "entry_idx": 0,
            "planned_exit_idx": 1,
            "exit_idx": 1,
            "exit_delay_steps": 0,
        }
    ]
    old_result = (
        {"avg_gross_exposure": 1.0},
        pd.Series([0.0]),
        pd.Series([0.0]),
        pd.Series([0.0]),
        period_info,
    )

    rebuilt = rebuild_backtest_from_positions(
        positions,
        old_result,
        context={
            "cash_gross_overlay": {"enabled": True},
            "post_buffer_exposure_repair": {"enabled": False},
            "backtest_long_only": True,
            "execution_model": build_execution_model(
                {},
                default_cost_bps=0.0,
                default_exit_price_policy="strict",
                default_exit_fallback_policy="ffill",
                default_price_col="close",
            ),
            "backtest_pricing_df": pricing,
            "backtest_cost_bps_effective": 0.0,
            "backtest_trading_days_per_year": 252,
            "backtest_exit_price_policy": "strict",
            "backtest_exit_fallback_policy": "ffill",
            "backtest_tradable_col": "is_tradable",
        },
    )

    stats, net_series, gross_series, turnover_series, periods = rebuilt
    assert stats["avg_gross_exposure"] == pytest.approx(0.90)
    assert stats["avg_cash_weight"] == pytest.approx(0.10)
    assert net_series.iloc[0] == pytest.approx(0.018)
    assert gross_series.iloc[0] == pytest.approx(0.018)
    assert turnover_series.iloc[0] == pytest.approx(0.90)
    assert periods[0]["gross_exposure"] == pytest.approx(0.90)
