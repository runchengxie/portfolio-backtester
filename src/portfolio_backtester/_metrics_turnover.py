"""Turnover estimation from ranked predictions across rebalance dates."""

from __future__ import annotations

import pandas as pd

from portfolio_backtester._symbol_utils import canonicalize_symbol_columns

from .portfolio_selection import apply_rank_offset, apply_rebalance_buffer


def estimate_turnover(
    data: pd.DataFrame,
    pred_col: str,
    k: int,
    rebalance_dates: list[pd.Timestamp],
    buffer_exit: int = 0,
    buffer_entry: int = 0,
    rank_offset: int = 0,
) -> pd.Series:
    if data is None or data.empty:
        return pd.Series(dtype=float, name="turnover")
    data = canonicalize_symbol_columns(data, context="Turnover data")
    prev = None
    turnovers: list[tuple[pd.Timestamp, float]] = []
    day_groups = {  # noqa: C416 - avoid relying on a shadowable dict() callable here.
        date: group for date, group in data.groupby("trade_date", sort=False)
    }
    for date in rebalance_dates:
        day = day_groups.get(date)
        if day is None or len(day) < k:
            continue
        ranked = apply_rank_offset(
            day.sort_values(pred_col, ascending=False)["symbol"].tolist(),
            rank_offset,
        )
        k_final = min(k, len(ranked))
        if k_final <= 0:
            continue
        holdings = set(
            apply_rebalance_buffer(
                ranked,
                prev,
                k_final,
                buffer_exit,
                buffer_entry,
            )[:k_final]
        )
        if prev is not None:
            overlap = len(holdings & prev)
            turnovers.append((pd.to_datetime(date), 1 - overlap / k_final))
        prev = holdings
    if not turnovers:
        return pd.Series(dtype=float, name="turnover")
    turnovers.sort(key=lambda x: x[0])
    return pd.Series(
        [value for _, value in turnovers],
        index=pd.Index([date for date, _ in turnovers], name="trade_date"),
        name="turnover",
    )
