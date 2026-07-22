"""Fail-closed input contract for research staggered-cohort execution."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import Any, cast

import numpy as np
import pandas as pd

from .execution_calendar import build_execution_date_map

_REQUIRED_RAW_PRICING_COLUMNS = {
    "trade_date",
    "symbol",
    "open",
    "up_limit",
    "down_limit",
    "is_suspended",
}
_SHANGHAI = "Asia/Shanghai"


@dataclass(frozen=True)
class StaggeredTarget:
    """One signal date mapped to its immutable next-open target."""

    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    entry_idx: int
    planned_exit_date: pd.Timestamp | None
    planned_exit_idx: int
    cohort_id: int
    symbols: tuple[str, ...]
    scores: tuple[float, ...]


def _open_calendar_dates(trade_calendar: pd.DataFrame | pd.DatetimeIndex) -> list[pd.Timestamp]:
    if isinstance(trade_calendar, pd.DatetimeIndex):
        dates = pd.DatetimeIndex(trade_calendar)
    else:
        required = {"cal_date", "is_open"}
        missing = sorted(required - set(trade_calendar.columns))
        if missing:
            raise ValueError(f"trade_calendar is missing columns: {missing}")
        open_flag = pd.to_numeric(trade_calendar["is_open"], errors="coerce")
        if open_flag.isna().any() or not open_flag.isin([0, 1]).all():
            raise ValueError("trade_calendar is_open must contain only zero or one")
        dates = pd.DatetimeIndex(
            pd.to_datetime(trade_calendar.loc[open_flag.eq(1), "cal_date"], errors="coerce")
        )
    if dates.isna().any():
        raise ValueError("trade_calendar contains invalid open dates")
    if dates.tz is not None:
        dates = dates.tz_localize(None)
    normalized = pd.Series(dates).dt.normalize().drop_duplicates().sort_values()
    if len(normalized) < 2:
        raise ValueError("trade_calendar must contain at least two open sessions")
    return cast(list[pd.Timestamp], normalized.tolist())


def prepare_staggered_pricing(
    pricing: pd.DataFrame,
    trade_calendar: pd.DataFrame | pd.DatetimeIndex,
    *,
    valuation_price_col: str = "open",
) -> tuple[pd.DataFrame, list[pd.Timestamp]]:
    """Validate raw tradability and valuation prices against an open calendar."""

    valuation_price_col = str(valuation_price_col).strip()
    if not valuation_price_col:
        raise ValueError("valuation_price_col must be a non-empty column name")
    required = {*_REQUIRED_RAW_PRICING_COLUMNS, valuation_price_col}
    missing = sorted(required - set(pricing.columns))
    if missing:
        raise ValueError(f"pricing is missing required execution columns: {missing}")
    work = pricing.loc[:, sorted(required)].copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"], errors="coerce").dt.normalize()
    work["symbol"] = work["symbol"].astype("string").str.strip()
    if work["trade_date"].isna().any() or work["symbol"].isna().any():
        raise ValueError("pricing contains invalid trade_date or symbol")
    if work.duplicated(["trade_date", "symbol"]).any():
        raise ValueError("pricing must be unique by trade_date and symbol")

    observed = pd.DatetimeIndex(work["trade_date"].unique()).sort_values()
    if observed.empty:
        raise ValueError("pricing must contain at least one observed open session")
    authoritative = _open_calendar_dates(trade_calendar)
    authoritative_set = set(authoritative)
    outside = sorted(set(cast(list[pd.Timestamp], observed.tolist())) - authoritative_set)
    if outside:
        raise ValueError(f"pricing contains dates outside the open calendar: {outside[:3]}")
    first, last = cast(pd.Timestamp, observed[0]), cast(pd.Timestamp, observed[-1])
    expected = [date for date in authoritative if first <= date <= last]
    missing_sessions = sorted(set(expected) - set(cast(list[pd.Timestamp], observed.tolist())))
    if missing_sessions:
        raise ValueError(
            f"pricing has whole-session gaps in the authoritative calendar: {missing_sessions[:3]}"
        )
    return work.set_index(["trade_date", "symbol"]).sort_index(), expected


def _aware_timestamp(value: object, *, field: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a timezone-aware timestamp") from exc
    if not isinstance(timestamp, pd.Timestamp) or timestamp.tzinfo is None:
        raise ValueError(f"{field} must be a timezone-aware timestamp")
    return timestamp.tz_convert(_SHANGHAI)


def _validate_availability(
    group: pd.DataFrame,
    *,
    signal_date: pd.Timestamp,
    entry_date: pd.Timestamp,
    available_at_col: str,
) -> None:
    available = [
        _aware_timestamp(value, field=available_at_col) for value in group[available_at_col]
    ]
    entry_open = entry_date.tz_localize(_SHANGHAI) + pd.Timedelta(hours=9, minutes=30)
    for timestamp in available:
        if timestamp.normalize().tz_localize(None) != signal_date:
            raise ValueError("signal available_at must fall on its T-close signal date")
        if timestamp.time() < time(15, 0):
            raise ValueError("signal available_at must be at or after the T close")
        if timestamp >= entry_open:
            raise ValueError("signal available_at must be strictly before the T+1 open")


def prepare_staggered_targets(
    signals: pd.DataFrame,
    trade_dates: list[pd.Timestamp],
    *,
    horizon_days: int,
    top_n: int,
    score_col: str,
    signal_date_col: str,
    available_at_col: str,
    allow_cash_shortfall: bool = False,
) -> list[StaggeredTarget]:
    """Freeze targets only after proving T-close/T+1 availability.

    When ``allow_cash_shortfall`` is true, a signal date may provide fewer
    than ``top_n`` names.  The missing fixed-size slots remain in cash rather
    than being redistributed across the selected names.
    """

    required = {signal_date_col, "symbol", score_col, available_at_col}
    missing = sorted(required - set(signals.columns))
    if missing:
        raise ValueError(f"signals is missing required columns: {missing}")
    work = signals.loc[:, sorted(required)].copy()
    work[signal_date_col] = pd.to_datetime(work[signal_date_col], errors="coerce").dt.normalize()
    work["symbol"] = work["symbol"].astype("string").str.strip()
    work[score_col] = pd.to_numeric(work[score_col], errors="coerce")
    if work[[signal_date_col, "symbol", score_col]].isna().any().any():
        raise ValueError("signals contains invalid dates, symbols, or scores")
    if not np.isfinite(work[score_col].to_numpy(dtype=float)).all():
        raise ValueError("signals scores must be finite")
    if work.duplicated([signal_date_col, "symbol"]).any():
        raise ValueError("signals must be unique by signal date and symbol")
    date_set = set(trade_dates)
    signal_dates = sorted(cast(list[pd.Timestamp], work[signal_date_col].unique().tolist()))
    if not signal_dates or any(date not in date_set for date in signal_dates):
        raise ValueError("every signal date must be present in the authoritative calendar")
    entry_map = build_execution_date_map(
        signal_dates,
        1,
        trade_dates,
        calendar="market",
        market="cn",
        allow_future=False,
    )
    if len(entry_map) != len(signal_dates):
        raise ValueError("every T-close signal requires an observed T+1 open")
    date_to_idx = {date: idx for idx, date in enumerate(trade_dates)}
    first_entry_idx = min(date_to_idx[date] for date in entry_map.values())
    targets: list[StaggeredTarget] = []
    for signal_date in signal_dates:
        group = work.loc[work[signal_date_col].eq(signal_date)].sort_values(
            [score_col, "symbol"], ascending=[False, True], kind="mergesort"
        )
        if len(group) < top_n and not allow_cash_shortfall:
            raise ValueError(f"signal date {signal_date.date()} has fewer than top_n candidates")
        entry_date = entry_map[signal_date]
        _validate_availability(
            group,
            signal_date=signal_date,
            entry_date=entry_date,
            available_at_col=available_at_col,
        )
        selected = group.head(top_n)
        entry_idx = date_to_idx[entry_date]
        exit_idx = entry_idx + horizon_days
        targets.append(
            StaggeredTarget(
                signal_date=signal_date,
                entry_date=entry_date,
                entry_idx=entry_idx,
                planned_exit_date=trade_dates[exit_idx] if exit_idx < len(trade_dates) else None,
                planned_exit_idx=exit_idx,
                cohort_id=(entry_idx - first_entry_idx) % horizon_days,
                symbols=tuple(selected["symbol"].astype(str)),
                scores=tuple(selected[score_col].astype(float)),
            )
        )
    return targets


__all__ = [
    "StaggeredTarget",
    "prepare_staggered_pricing",
    "prepare_staggered_targets",
]
