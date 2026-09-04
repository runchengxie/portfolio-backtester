"""Reusable helpers for deterministic backtest grid commands."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pandas as pd

from .rebalance import get_rebalance_dates


def resolve_output_path(path_text: str) -> Path:
    """Resolve a grid output path relative to the caller's working directory."""
    candidate = Path(path_text).expanduser()
    return candidate if candidate.is_absolute() else (Path.cwd() / candidate).resolve()


def safe_run_name(
    base: str,
    top_k: int,
    cost_bps: float,
    *,
    buffer_exit: int,
    buffer_entry: int,
    include_buffer: bool,
    weighting: str,
    include_weighting: bool,
) -> str:
    cost_text = f"{cost_bps:g}".replace(".", "p")
    run_name = f"{base}_k{top_k}_bps{cost_text}"
    if include_buffer:
        run_name = f"{run_name}_bx{int(buffer_exit)}_be{int(buffer_entry)}"
    if include_weighting:
        run_name = f"{run_name}_w{weighting}"
    return run_name


def parse_date_list(values: list[str] | None) -> list[pd.Timestamp]:
    """Parse, sort, and de-duplicate compact or ISO date values."""
    if not values:
        return []
    parsed_dates: set[pd.Timestamp] = set()
    for raw in values:
        text = str(raw).strip() if raw is not None else ""
        if not text:
            continue
        parsed = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
        if pd.isna(parsed):
            parsed = pd.to_datetime(text, errors="coerce")
        if pd.notna(parsed):
            parsed_dates.add(cast(pd.Timestamp, parsed))
    return sorted(parsed_dates)


def resolve_rebalance_dates(
    summary_dates: list[str] | None,
    scored_data: pd.DataFrame,
    frequency: str,
    min_symbols_per_date: int,
) -> list[pd.Timestamp]:
    """Resolve explicit dates or derive a validated rebalance schedule."""
    parsed = parse_date_list(summary_dates)
    if parsed:
        available = set(pd.to_datetime(scored_data["trade_date"].unique()))
        return [date for date in parsed if date in available]

    trade_dates = sorted(pd.to_datetime(scored_data["trade_date"].unique()))
    rebalance_dates = get_rebalance_dates(trade_dates, frequency)
    if min_symbols_per_date > 1:
        counts = scored_data.groupby("trade_date")["symbol"].nunique()
        valid_dates = set(counts[counts >= min_symbols_per_date].index)
        rebalance_dates = [date for date in rebalance_dates if date in valid_dates]
    return rebalance_dates


__all__ = ["parse_date_list", "resolve_output_path", "resolve_rebalance_dates", "safe_run_name"]
