import pandas as pd
import pytest

from portfolio_backtester.reporting import (
    build_backtest_layer_comparison_frame,
    build_backtest_report,
    build_benchmark_compare_summary_frame,
)


def test_build_backtest_report_includes_relative_nav_and_rolling_metrics():
    dates = pd.date_range("2020-01-31", periods=72, freq="ME")
    strategy = pd.Series(0.01, index=dates, name="strategy_return")
    benchmark = pd.Series(0.005, index=dates, name="benchmark_return")

    report = build_backtest_report(
        strategy_returns=strategy,
        benchmark_returns=benchmark,
        periods_per_year=12.0,
    )

    assert report.index.name == "trade_date"
    assert {
        "strategy_return",
        "strategy_nav",
        "benchmark_return",
        "benchmark_nav",
        "active_return",
        "relative_nav",
        "strategy_rolling_cagr_1y",
        "strategy_rolling_cagr_3y",
        "strategy_rolling_cagr_5y",
        "strategy_rolling_max_drawdown_1y",
        "strategy_rolling_max_drawdown_3y",
        "strategy_rolling_max_drawdown_5y",
    }.issubset(report.columns)
    assert report["relative_nav"].iloc[-1] == pytest.approx(
        report["strategy_nav"].iloc[-1] / report["benchmark_nav"].iloc[-1]
    )
    assert report["strategy_rolling_cagr_1y"].iloc[-1] == pytest.approx((1.01**12) - 1.0)
    assert report["strategy_rolling_max_drawdown_5y"].iloc[-1] == pytest.approx(0.0)


def test_build_backtest_layer_comparison_frame_flattens_accounting_layers():
    frame = build_backtest_layer_comparison_frame(
        strategy_stats={
            "periods": 12,
            "periods_per_year": 12.0,
            "total_return": 0.12,
            "ann_return": 0.12,
            "ann_vol": 0.10,
            "sharpe": 1.2,
            "max_drawdown": -0.05,
        },
        ideal_daily_nav_summary={
            "status": "ok",
            "daily_rows": 252,
            "fill_ratio": 1.0,
            "stats": {
                "periods": 252,
                "periods_per_year": 252.0,
                "total_return": 0.10,
                "ann_return": 0.10,
                "ann_vol": 0.12,
                "sharpe": 0.9,
                "max_drawdown": -0.08,
            },
        },
        execution_sim_executed_summary={
            "status": "ok",
            "daily_rows": 252,
            "fill_ratio": 0.95,
            "unfilled_notional": 1000.0,
            "stats": {
                "periods": 252,
                "periods_per_year": 252.0,
                "total_return": 0.08,
                "ann_return": 0.08,
                "ann_vol": 0.13,
                "sharpe": 0.7,
                "max_drawdown": -0.09,
            },
        },
    )

    assert frame["layer"].tolist() == [
        "core_period_return",
        "ideal_daily_nav",
        "execution_sim.executed",
    ]
    ideal = frame.set_index("layer").loc["ideal_daily_nav"]
    executed = frame.set_index("layer").loc["execution_sim.executed"]
    assert ideal["daily_rows"] == 252
    assert ideal["fill_ratio"] == pytest.approx(1.0)
    assert executed["total_return"] == pytest.approx(0.08)
    assert executed["unfilled_notional"] == pytest.approx(1000.0)


def test_build_benchmark_compare_summary_frame_flattens_metrics():
    summary = build_benchmark_compare_summary_frame(
        [
            {
                "name": "hk_02800",
                "source_type": "returns_file",
                "returns_file": "/tmp/hk_02800.csv",
                "symbol": None,
                "is_primary": False,
                "aligned_periods": 24,
                "benchmark": {
                    "ann_return": 0.12,
                    "ann_vol": 0.2,
                    "sharpe": 0.6,
                    "max_drawdown": -0.18,
                    "total_return": 0.25,
                },
                "active": {
                    "tracking_error": 0.11,
                    "information_ratio": 0.4,
                    "beta": 0.9,
                    "alpha": 0.02,
                    "corr": 0.7,
                    "active_total_return": 0.03,
                },
                "report_file": "/tmp/report.csv",
            }
        ]
    )

    row = summary.iloc[0]
    assert row["name"] == "hk_02800"
    assert row["source_type"] == "returns_file"
    assert row["benchmark_ann_return"] == pytest.approx(0.12)
    assert row["active_information_ratio"] == pytest.approx(0.4)
    assert row["report_file"] == "/tmp/report.csv"
