from __future__ import annotations

import numpy as np
import pandas as pd

from cstree.backtesting.rebalance import estimate_rebalance_gap, get_rebalance_dates


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
