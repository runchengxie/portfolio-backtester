"""Portfolio turnover accounting with price-induced weight drift."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from .portfolio_weights import normalize_position_weights


def compute_trade_summary(
    prev_weights: pd.Series | None,
    prev_prices: pd.Series | None,
    prev_date: pd.Timestamp | None,
    target_weights: pd.Series,
    entry_date: pd.Timestamp,
    *,
    price_table: pd.DataFrame,
) -> tuple[float, float, float, pd.Series]:
    """Return total, entry and exit turnover plus signed trade weights."""

    if target_weights is None or target_weights.empty:
        return 0.0, 0.0, 0.0, pd.Series(dtype=float)

    target_clean = normalize_position_weights(target_weights)
    if target_clean.empty:
        return 0.0, 0.0, 0.0, pd.Series(dtype=float)

    if prev_weights is None or prev_weights.empty:
        trade_weights = target_clean.copy()
        traded = float(trade_weights.abs().sum())
        return traded, traded, 0.0, trade_weights

    prev_clean = normalize_position_weights(prev_weights)
    drift_weights = drift_previous_weights(
        prev_clean,
        prev_prices,
        prev_date,
        entry_date,
        price_table=price_table,
    )
    all_ids = drift_weights.index.union(target_clean.index)
    drift_aligned = drift_weights.reindex(all_ids).fillna(0.0)
    target_aligned = target_clean.reindex(all_ids).fillna(0.0)
    trade_weights = target_aligned - drift_aligned
    entry_turnover = float(trade_weights.clip(lower=0.0).sum())
    exit_turnover = float((-trade_weights.clip(upper=0.0)).sum())
    turnover = 0.5 * float(np.abs(trade_weights).sum())
    return turnover, entry_turnover, exit_turnover, trade_weights


def drift_previous_weights(
    prev_clean: pd.Series,
    prev_prices: pd.Series | None,
    prev_date: pd.Timestamp | None,
    entry_date: pd.Timestamp,
    *,
    price_table: pd.DataFrame,
) -> pd.Series:
    """Drift normalized prior weights to entry-date prices when prices are available."""

    if prev_prices is None or prev_date is None:
        return prev_clean
    prev_prices_valid = prev_prices.reindex(prev_clean.index)
    prev_prices_valid = cast(pd.Series, prev_prices_valid[prev_prices_valid.notna()])
    if prev_prices_valid.empty or entry_date not in price_table.index:
        return prev_clean
    prev_clean = prev_clean.reindex(prev_prices_valid.index).dropna()
    current_prices = cast(pd.Series, price_table.loc[entry_date, prev_prices_valid.index])
    valid_prev = current_prices.notna()
    prev_prices_valid = cast(pd.Series, prev_prices_valid[valid_prev])
    current_prices = current_prices[valid_prev]
    prev_clean = prev_clean.reindex(prev_prices_valid.index).dropna()
    if prev_prices_valid.empty or prev_clean.empty:
        return prev_clean
    drift = prev_clean * (current_prices / prev_prices_valid)
    if float(drift.sum()) <= 0:
        return prev_clean
    return normalize_position_weights(drift)


# Compatibility aliases for callers migrating private orchestration helpers.
_compute_trade_summary = compute_trade_summary
_drift_previous_weights = drift_previous_weights


__all__ = [
    "_compute_trade_summary",
    "_drift_previous_weights",
    "compute_trade_summary",
    "drift_previous_weights",
]
