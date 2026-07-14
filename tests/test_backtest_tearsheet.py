from datetime import datetime

import pandas as pd

from portfolio_backtester.tearsheet import (
    build_backtest_tearsheet_html,
    write_backtest_tearsheet,
)


def test_build_backtest_tearsheet_html_includes_core_sections_and_escapes_title():
    dates = pd.date_range("2020-01-31", periods=18, freq="ME")
    strategy = pd.Series(
        [
            0.02,
            -0.04,
            0.03,
            0.01,
            -0.02,
            0.04,
            0.02,
            -0.03,
            0.05,
            0.01,
            -0.01,
            0.03,
            0.02,
            -0.02,
            0.04,
            0.01,
            -0.03,
            0.02,
        ],
        index=dates,
        name="strategy_return",
    )
    benchmark = pd.Series(0.005, index=dates, name="benchmark_return")
    daily_dates = pd.date_range("2020-01-31", periods=30, freq="B")
    ideal_daily = pd.DataFrame(
        {
            "trade_date": daily_dates,
            "executed_nav": 1.0 + pd.Series(range(30), dtype=float) * 0.001,
        }
    )
    executed_daily = pd.DataFrame(
        {
            "trade_date": daily_dates,
            "executed_nav": 1.0 + pd.Series(range(30), dtype=float) * 0.0008,
        }
    )

    html = build_backtest_tearsheet_html(
        strategy_returns=strategy,
        strategy_stats={"periods_per_year": 12.0},
        benchmark_returns=benchmark,
        benchmark_stats={"periods_per_year": 12.0},
        active_stats={"tracking_error": 0.05, "information_ratio": 0.3},
        title="HK <Alpha>",
        benchmark_name="02800.HK",
        generated_at=datetime(2026, 5, 10, 9, 30, 0),
        ideal_daily_nav_summary={
            "status": "ok",
            "fill_ratio": 1.0,
            "stats": {
                "periods": 30,
                "total_return": 0.03,
                "sharpe": 0.8,
                "max_drawdown": -0.04,
            },
        },
        ideal_daily_nav_daily=ideal_daily,
        execution_sim_executed_summary={
            "status": "ok",
            "fill_ratio": 0.95,
            "stats": {
                "periods": 30,
                "total_return": 0.024,
                "sharpe": 0.7,
                "max_drawdown": -0.05,
            },
        },
        execution_sim_executed_daily=executed_daily,
    )

    assert "HK &lt;Alpha&gt;" in html
    assert "Benchmark: 02800.HK" in html
    assert "Generated: 2026-05-10 09:30:00" in html
    assert "Cumulative Returns vs Benchmark" in html
    assert "Underwater Plot" in html
    assert "Backtest Layer NAV Comparison" in html
    assert "Strategy - Monthly Returns (%)" in html
    assert "Key Performance Metrics" in html
    assert "Backtest Accounting Layers" in html
    assert "Core period return" in html
    assert "Ideal daily NAV" in html
    assert "Execution-adjusted NAV" in html
    assert "EOY Returns vs Benchmark" in html
    assert "Worst 10 Drawdowns" in html
    assert "<svg" in html


def test_write_backtest_tearsheet_writes_html_file(tmp_path):
    dates = pd.date_range("2021-01-31", periods=6, freq="ME")
    strategy = pd.Series([0.01, -0.02, 0.03, 0.01, -0.01, 0.02], index=dates)
    path = tmp_path / "backtest_tearsheet.html"

    write_backtest_tearsheet(
        path=path,
        strategy_returns=strategy,
        strategy_stats={"periods_per_year": 12.0},
        benchmark_returns=None,
        benchmark_stats=None,
        active_stats=None,
        title="Backtest",
        generated_at=datetime(2026, 5, 10, 9, 30, 0),
    )

    content = path.read_text(encoding="utf-8")
    assert content.startswith("<!DOCTYPE html>")
    assert "Cumulative Returns" in content
    assert "Benchmark:" not in content
