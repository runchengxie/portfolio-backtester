from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from portfolio_backtester import backtest_topk
from portfolio_backtester.rebalance import (
    SessionRebalanceSchedule,
    estimate_rebalance_gap,
    get_rebalance_dates,
    get_session_interval_rebalance_dates,
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


def test_session_interval_anchor_is_stable_when_history_is_extended() -> None:
    full = pd.bdate_range("2024-01-02", "2024-01-19")
    short = full[3:]
    anchor = pd.Timestamp(short[2])

    full_schedule = get_session_interval_rebalance_dates(
        full,
        rebalance_interval_sessions=3,
        anchor=anchor,
        phase=1,
    )
    short_schedule = get_session_interval_rebalance_dates(
        short,
        rebalance_interval_sessions=3,
        anchor=anchor,
        phase=1,
    )

    assert [date for date in full_schedule if date >= short[0]] == short_schedule


@pytest.mark.parametrize("interval", [3, 5])
def test_session_interval_phases_partition_the_trading_calendar(interval: int) -> None:
    sessions = pd.bdate_range("2024-01-02", periods=23)
    anchor = pd.Timestamp(sessions[7])

    phases = [
        get_session_interval_rebalance_dates(
            sessions,
            rebalance_interval_sessions=interval,
            anchor=anchor,
            phase=phase,
        )
        for phase in range(interval)
    ]

    flattened = [date for phase_dates in phases for date in phase_dates]
    assert sorted(flattened) == list(sessions)
    assert len(flattened) == len(set(flattened))
    session_index = {date: index for index, date in enumerate(sessions)}
    for phase_dates in phases:
        gaps = [session_index[right] - session_index[left] for left, right in pairwise(phase_dates)]
        assert gaps == [interval] * len(gaps)


def test_session_interval_schedule_declares_hold_until_next_rebalance() -> None:
    sessions = pd.bdate_range("2024-01-02", periods=10)
    schedule = SessionRebalanceSchedule(
        rebalance_interval_sessions=3,
        anchor=pd.Timestamp(sessions[0]),
        phase=0,
    )

    assert schedule.holding_mode == "until_next_rebalance"
    assert schedule.dates(sessions) == list(sessions[[0, 3, 6, 9]])


def test_session_interval_backtest_exits_only_at_next_scheduled_rebalance() -> None:
    sessions = pd.bdate_range("2024-01-02", periods=10)
    schedule = SessionRebalanceSchedule(3, pd.Timestamp(sessions[0]))
    rebalance_dates = schedule.dates(sessions)
    data = pd.DataFrame(
        {
            "trade_date": sessions,
            "symbol": "A",
            "score": 1.0,
            "close": range(10, 20),
        }
    )

    result = backtest_topk(
        data,
        pred_col="score",
        price_col="close",
        rebalance_dates=rebalance_dates,
        top_k=1,
        shift_days=0,
        cost_bps=0.0,
        trading_days_per_year=252,
    )

    assert result is not None
    periods = result[4]
    assert [period["entry_date"] for period in periods] == rebalance_dates[:-1]
    assert [period["planned_exit_date"] for period in periods] == rebalance_dates[1:]


def test_session_interval_schedule_rejects_non_session_anchor() -> None:
    sessions = pd.bdate_range("2024-01-02", periods=5)

    with pytest.raises(ValueError, match="anchor must be present"):
        get_session_interval_rebalance_dates(
            sessions,
            rebalance_interval_sessions=3,
            anchor=pd.Timestamp("2024-01-06"),
        )


@pytest.mark.parametrize("interval", [True, 3.0, "3", 0])
def test_session_interval_schedule_requires_positive_integer(interval: object) -> None:
    sessions = pd.bdate_range("2024-01-02", periods=5)

    with pytest.raises(ValueError, match="positive integer"):
        get_session_interval_rebalance_dates(
            sessions,
            rebalance_interval_sessions=interval,  # type: ignore[arg-type]
            anchor=pd.Timestamp(sessions[0]),
        )


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
