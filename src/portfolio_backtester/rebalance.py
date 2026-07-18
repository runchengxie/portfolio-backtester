from __future__ import annotations

import operator
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

_MULTIWEEK_PATTERN = re.compile(
    r"^(?P<base>BW|BIWEEKLY|BI-WEEKLY|FORTNIGHTLY|(?P<weeks>[2-9])W)"
    r"(?:-(?P<weekday>MON|TUE|WED|THU|FRI|SAT|SUN))?"
    r"(?:-(?P<phase>0|1|ALT))?$"
)
_ANCHORED_MULTIWEEK_PATTERN = re.compile(
    r"^(?P<base>BW|BIWEEKLY|BI-WEEKLY|FORTNIGHTLY|(?P<weeks>[2-9])W)"
    r"(?:-(?P<weekday>MON|TUE|WED|THU|FRI|SAT|SUN))?"
    r"(?:-(?P<phase>0|1|ALT))?"
    r"@(?P<anchor>\d{4}-?\d{2}-?\d{2})$"
)


@dataclass(frozen=True)
class _MultiweekFrequency:
    weekly_freq: str
    phase: int
    weeks: int
    anchor: pd.Timestamp | None = None


@dataclass(frozen=True)
class SessionRebalanceSchedule:
    """Deterministic every-N-trading-session rebalance schedule.

    ``anchor`` is a trading session in the supplied calendar.  Keeping it
    explicit makes a phase stable when callers extend the sample window.
    Positions are held until the next selected rebalance; this is deliberately
    different from a daily staggered sleeve labelled H3/H5.
    """

    rebalance_interval_sessions: int
    anchor: pd.Timestamp
    phase: int = 0
    holding_mode: Literal["until_next_rebalance"] = "until_next_rebalance"

    def __post_init__(self) -> None:
        interval = _positive_session_interval(self.rebalance_interval_sessions)
        phase = _session_phase(self.phase, interval=interval)
        anchor = pd.to_datetime(self.anchor, errors="coerce")
        if pd.isna(anchor):
            raise ValueError("anchor must be a valid trading-session date.")
        if self.holding_mode != "until_next_rebalance":
            raise ValueError("holding_mode must be 'until_next_rebalance'.")
        object.__setattr__(self, "rebalance_interval_sessions", interval)
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "anchor", pd.Timestamp(anchor).normalize())

    def dates(self, trading_sessions: Iterable[pd.Timestamp]) -> list[pd.Timestamp]:
        """Return the sessions selected by this anchored phase."""

        return get_session_interval_rebalance_dates(
            trading_sessions,
            rebalance_interval_sessions=self.rebalance_interval_sessions,
            anchor=self.anchor,
            phase=self.phase,
        )


def _positive_session_interval(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("rebalance_interval_sessions must be a positive integer.")
    try:
        normalized = operator.index(value)
    except TypeError as exc:
        raise ValueError("rebalance_interval_sessions must be a positive integer.") from exc
    if normalized <= 0:
        raise ValueError("rebalance_interval_sessions must be a positive integer.")
    return int(normalized)


def _session_phase(value: object, *, interval: int) -> int:
    if isinstance(value, bool):
        raise ValueError("phase must be an integer in [0, rebalance_interval_sessions).")
    try:
        normalized = operator.index(value)
    except TypeError as exc:
        raise ValueError("phase must be an integer in [0, rebalance_interval_sessions).") from exc
    if not 0 <= normalized < interval:
        raise ValueError("phase must be an integer in [0, rebalance_interval_sessions).")
    return int(normalized)


def get_session_interval_rebalance_dates(
    trading_sessions: Iterable[pd.Timestamp],
    *,
    rebalance_interval_sessions: int,
    anchor: pd.Timestamp,
    phase: int = 0,
) -> list[pd.Timestamp]:
    """Select one deterministic phase from an ordered trading-session calendar.

    The anchor must be present in ``trading_sessions``.  Requiring an exact
    session prevents a calendar date from silently changing phase after a
    holiday or when a caller trims the input window.
    """

    sessions = _clean_dates(trading_sessions)
    interval = _positive_session_interval(rebalance_interval_sessions)
    normalized_phase = _session_phase(phase, interval=interval)
    normalized_anchor = pd.to_datetime(anchor, errors="coerce")
    if pd.isna(normalized_anchor):
        raise ValueError("anchor must be a valid trading-session date.")
    anchor_session = pd.Timestamp(normalized_anchor).normalize()
    if anchor_session not in sessions:
        raise ValueError("anchor must be present in trading_sessions.")
    anchor_index = sessions.index(anchor_session)
    return [
        session
        for index, session in enumerate(sessions)
        if (index - anchor_index - normalized_phase) % interval == 0
    ]


def _clean_dates(dates: Iterable[pd.Timestamp]) -> list[pd.Timestamp]:
    date_series = pd.to_datetime(pd.Series(list(dates), name="date"), errors="coerce")
    return sorted(pd.Timestamp(date).normalize() for date in date_series.dropna().unique())


def _parse_anchor(value: str) -> pd.Timestamp:
    anchor = pd.to_datetime(value, format="%Y%m%d" if "-" not in value else None)
    return pd.Timestamp(anchor).normalize()


def _parse_multiweek_frequency(freq: str) -> _MultiweekFrequency | None:
    normalized = str(freq or "").strip().upper()
    match = _ANCHORED_MULTIWEEK_PATTERN.match(normalized) or _MULTIWEEK_PATTERN.match(normalized)
    if match is None:
        return None
    weekday = match.group("weekday")
    phase_token = match.group("phase")
    weeks = int(match.group("weeks") or 2)
    phase = 1 if phase_token in {"1", "ALT"} else 0
    weekly_freq = f"W-{weekday}" if weekday else "W"
    anchor_token = match.groupdict().get("anchor")
    anchor = _parse_anchor(anchor_token) if anchor_token else None
    return _MultiweekFrequency(
        weekly_freq=weekly_freq,
        phase=phase,
        weeks=weeks,
        anchor=anchor,
    )


def _period_end_dates(dates: list[pd.Timestamp], freq: str) -> list[pd.Timestamp]:
    date_df = pd.DataFrame({"date": dates})
    date_df["period"] = date_df["date"].dt.to_period(freq)
    return date_df.groupby("period")["date"].max().sort_values().tolist()


def _multiweek_rebalance_dates(
    dates: list[pd.Timestamp],
    weekly_freq: str,
    phase: int,
    weeks: int,
    anchor: pd.Timestamp | None = None,
) -> list[pd.Timestamp]:
    date_df = pd.DataFrame({"date": dates})
    date_df["period"] = date_df["date"].dt.to_period(weekly_freq)
    weekly_dates = date_df.groupby("period")["date"].max().sort_index()
    if anchor is None:
        return weekly_dates.tolist()[phase::weeks]

    anchor_ordinal = anchor.to_period(weekly_freq).ordinal
    selected_dates: list[pd.Timestamp] = []
    for period, date in weekly_dates.items():
        if (period.ordinal - anchor_ordinal - phase) % weeks == 0:
            selected_dates.append(pd.Timestamp(date))
    return selected_dates


def get_rebalance_dates(dates: Iterable[pd.Timestamp], freq: str) -> list[pd.Timestamp]:
    """Return rebalance dates based on a pandas Period frequency.

    Biweekly aliases use every other weekly rebalance date. Pandas Period does
    not make ``2W`` coarser than ``W``, so this function handles those aliases
    before falling back to generic Period grouping. Multiweek frequencies can
    add an anchor suffix such as ``2W-FRI@20240419`` so phase selection is
    stable when the available date range changes.
    """
    dates_list = _clean_dates(dates)
    if not dates_list:
        return []
    if not freq or str(freq).upper() == "D":
        return dates_list

    multiweek = _parse_multiweek_frequency(freq)
    if multiweek is not None:
        return _multiweek_rebalance_dates(
            dates_list,
            multiweek.weekly_freq,
            multiweek.phase,
            multiweek.weeks,
            multiweek.anchor,
        )

    return _period_end_dates(dates_list, freq)


def _timestamp_set(dates: Iterable[pd.Timestamp]) -> set[pd.Timestamp]:
    return {pd.Timestamp(date) for date in pd.to_datetime(list(dates))}


def sample_rebalance_frame(
    frame: pd.DataFrame | None,
    *,
    frequency: str,
    valid_dates: Iterable[pd.Timestamp] | None = None,
    allowed_dates: Iterable[pd.Timestamp] | None = None,
) -> tuple[pd.DataFrame, list[pd.Timestamp]]:
    if frame is None or frame.empty:
        columns = frame.columns if frame is not None else pd.Index([])
        return pd.DataFrame(columns=columns), []

    frame_sorted = frame.sort_values("trade_date", kind="mergesort").reset_index(drop=True)
    normalized_trade_dates = pd.Series(
        pd.to_datetime(frame_sorted["trade_date"], errors="coerce"),
        index=frame_sorted.index,
    )
    trade_dates_sorted = sorted(
        pd.Timestamp(date) for date in normalized_trade_dates.dropna().unique()
    )
    rebalance_dates = get_rebalance_dates(trade_dates_sorted, frequency)
    if valid_dates:
        valid_dates_set = _timestamp_set(valid_dates)
        rebalance_dates = [date for date in rebalance_dates if date in valid_dates_set]
    if allowed_dates is not None:
        allowed_dates_set = _timestamp_set(allowed_dates)
        rebalance_dates = [date for date in rebalance_dates if date in allowed_dates_set]
    if not rebalance_dates:
        return frame_sorted.iloc[0:0].copy(), []

    sampled = frame_sorted.loc[normalized_trade_dates.isin(set(rebalance_dates))].copy()
    return sampled, rebalance_dates


_sample_rebalance_frame = sample_rebalance_frame


def estimate_rebalance_gap(
    trade_dates: Iterable[pd.Timestamp],
    rebalance_dates: Iterable[pd.Timestamp],
) -> float:
    trade_dates_sorted = list(pd.to_datetime(list(trade_dates)))
    rebalance_dates_sorted = list(pd.to_datetime(list(rebalance_dates)))
    if len(rebalance_dates_sorted) < 2 or len(trade_dates_sorted) < 2:
        return np.nan
    date_to_idx = {date: idx for idx, date in enumerate(sorted(trade_dates_sorted))}
    gaps: list[int] = []
    for i in range(len(rebalance_dates_sorted) - 1):
        start = rebalance_dates_sorted[i]
        end = rebalance_dates_sorted[i + 1]
        if start in date_to_idx and end in date_to_idx:
            gaps.append(date_to_idx[end] - date_to_idx[start])
    if not gaps:
        return np.nan
    median_gap = float(np.median(gaps))
    return float(np.floor(median_gap + 0.5))
