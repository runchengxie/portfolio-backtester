import logging

import pandas as pd
import pytest

from cstree.backtesting.benchmarking import build_benchmark_series, warn_if_delay_exit_lag


def test_build_benchmark_series_compounds_daily_returns_over_periods():
    benchmark_returns = pd.Series(
        [0.10, 0.01, 0.02, -0.03],
        index=pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06"]),
    )
    periods = [
        {"entry_date": pd.Timestamp("2020-01-01"), "exit_date": pd.Timestamp("2020-01-03")},
        {"entry_date": pd.Timestamp("2020-01-03"), "exit_date": pd.Timestamp("2020-01-06")},
    ]

    series, used_periods = build_benchmark_series(
        None,
        "open",
        "close",
        periods,
        benchmark_return_series=benchmark_returns,
    )

    assert series.index.tolist() == [pd.Timestamp("2020-01-03"), pd.Timestamp("2020-01-06")]
    assert series.iloc[0] == pytest.approx((1.01 * 1.02) - 1.0)
    assert series.iloc[1] == pytest.approx(-0.03)
    assert used_periods == periods


def test_build_benchmark_series_keeps_exit_dated_period_returns_compatible():
    benchmark_returns = pd.Series(
        [0.03, -0.01],
        index=pd.to_datetime(["2020-01-03", "2020-01-06"]),
    )
    periods = [
        {"entry_date": pd.Timestamp("2020-01-01"), "exit_date": pd.Timestamp("2020-01-03")},
        {"entry_date": pd.Timestamp("2020-01-03"), "exit_date": pd.Timestamp("2020-01-06")},
    ]

    series, used_periods = build_benchmark_series(
        None,
        "open",
        "close",
        periods,
        benchmark_return_series=benchmark_returns,
    )

    assert series.tolist() == pytest.approx([0.03, -0.01])
    assert used_periods == periods


def test_warn_if_delay_exit_lag_emits_warning(caplog):
    caplog.set_level(logging.WARNING, logger="cstree")

    warn_if_delay_exit_lag(
        label_prefix="[wf] ",
        exit_price_policy="delay",
        stats={
            "periods": 5,
            "periods_with_delayed_exit": 2,
            "avg_exit_lag_days": 1.5,
            "max_exit_lag_days": 3,
        },
    )

    assert "Delay exit policy produced lagged exits in 2/5 periods" in caplog.text


def test_warn_if_delay_exit_lag_skips_non_delay_policy(caplog):
    caplog.set_level(logging.WARNING, logger="cstree")

    warn_if_delay_exit_lag(
        label_prefix="[wf] ",
        exit_price_policy="same_day",
        stats={"periods": 5, "periods_with_delayed_exit": 2},
    )

    assert caplog.text == ""
