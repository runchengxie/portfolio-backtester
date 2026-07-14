from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio_backtester.rebalance import (
    estimate_rebalance_gap,
    get_rebalance_dates,
    sample_rebalance_frame,
)


def test_get_rebalance_dates_month_end() -> None:
    dates = pd.to_datetime(["2020-01-02", "2020-01-15", "2020-01-31", "2020-02-03", "2020-02-28"])

    rebal = get_rebalance_dates(dates, "M")

    assert rebal == [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-28")]


def test_get_rebalance_dates_biweekly_uses_every_other_week() -> None:
    dates = pd.bdate_range("2020-01-01", "2020-02-14")

    rebal = get_rebalance_dates(dates, "2W")

    assert (
        rebal
        == pd.to_datetime(
            [
                "2020-01-03",
                "2020-01-17",
                "2020-01-31",
                "2020-02-14",
            ]
        ).tolist()
    )


def test_get_rebalance_dates_biweekly_alt_phase() -> None:
    dates = pd.bdate_range("2020-01-01", "2020-02-14")

    rebal = get_rebalance_dates(dates, "BW-ALT")

    assert (
        rebal
        == pd.to_datetime(
            [
                "2020-01-10",
                "2020-01-24",
                "2020-02-07",
            ]
        ).tolist()
    )


def test_get_rebalance_dates_biweekly_anchor_keeps_calendar_phase() -> None:
    dates = pd.bdate_range("2020-01-08", "2020-02-14")

    rebal = get_rebalance_dates(dates, "2W-FRI@20200103")

    assert (
        rebal
        == pd.to_datetime(
            [
                "2020-01-17",
                "2020-01-31",
                "2020-02-14",
            ]
        ).tolist()
    )


def test_get_rebalance_dates_biweekly_anchor_supports_alt_phase() -> None:
    dates = pd.bdate_range("2020-01-01", "2020-02-14")

    rebal = get_rebalance_dates(dates, "2W-FRI-ALT@20200103")

    assert (
        rebal
        == pd.to_datetime(
            [
                "2020-01-10",
                "2020-01-24",
                "2020-02-07",
            ]
        ).tolist()
    )


def test_get_rebalance_dates_four_week_anchor_samples_every_fourth_week() -> None:
    dates = pd.bdate_range("2024-04-01", "2024-06-30")

    rebal = get_rebalance_dates(dates, "4W-FRI@20240419")

    assert [date.strftime("%Y%m%d") for date in rebal] == [
        "20240419",
        "20240517",
        "20240614",
    ]


def test_three_week_rebalance_alias_samples_every_third_week() -> None:
    dates = pd.bdate_range("2024-01-01", "2024-03-31")

    result = get_rebalance_dates(dates, "3W-FRI")

    assert [date.strftime("%Y%m%d") for date in result[:5]] == [
        "20240105",
        "20240126",
        "20240216",
        "20240308",
        "20240329",
    ]


def test_estimate_rebalance_gap_median() -> None:
    trade_dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06"])
    rebalance_dates = pd.to_datetime(["2020-01-01", "2020-01-03", "2020-01-06"])

    gap = estimate_rebalance_gap(trade_dates, rebalance_dates)

    assert np.isclose(gap, 2.0)


def test_sample_rebalance_frame_sorts_and_filters_dates() -> None:
    frame = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(
                [
                    "2024-01-12",
                    "2024-01-05",
                    "2024-01-19",
                    "2024-01-12",
                ]
            ),
            "symbol": ["CCC", "AAA", "DDD", "BBB"],
            "weight": [0.3, 0.1, 0.4, 0.2],
        }
    )

    sampled, rebalance_dates = sample_rebalance_frame(
        frame,
        frequency="W",
        valid_dates={pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-12")},
        allowed_dates=pd.DatetimeIndex(["2024-01-12", "2024-01-19"]),
    )

    assert rebalance_dates == [pd.Timestamp("2024-01-12")]
    assert sampled["symbol"].tolist() == ["CCC", "BBB"]


def test_sample_rebalance_frame_handles_empty_input() -> None:
    frame = pd.DataFrame(columns=["trade_date", "symbol"])

    sampled, rebalance_dates = sample_rebalance_frame(frame, frequency="W")

    assert sampled.empty
    assert sampled.columns.tolist() == ["trade_date", "symbol"]
    assert rebalance_dates == []
