"""Canonical turnover definitions shared by portfolio backtests.

The project historically used the word ``turnover`` for both holding-name
replacement and weight turnover. This module keeps those concepts explicit
and records the buy, sell, and gross traded weights needed for transaction-cost
accounting.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TurnoverBreakdown:
    """Weight turnover for one rebalance.

    ``half_l1_turnover`` is always ``0.5 * sum(abs(delta_weight))``.
    ``one_way_turnover`` preserves the backtester's historical convention for
    the initial portfolio: the initial buy is charged at full gross exposure,
    while subsequent rebalances use half-L1 turnover.
    """

    buy_weight: float
    sell_weight: float
    gross_traded_weight: float
    half_l1_turnover: float
    one_way_turnover: float
    is_initial: bool = False

    def to_dict(self) -> dict[str, float | bool]:
        return {
            "buy_weight": self.buy_weight,
            "sell_weight": self.sell_weight,
            "gross_traded_weight": self.gross_traded_weight,
            "half_l1_turnover": self.half_l1_turnover,
            "one_way_turnover": self.one_way_turnover,
            "is_initial": self.is_initial,
        }


def turnover_from_trade_weights(
    trade_weights: pd.Series | None,
    *,
    is_initial: bool = False,
) -> TurnoverBreakdown:
    """Return canonical turnover fields from signed trade weights.

    Positive values are buys and negative values are sells. Missing and
    non-finite values are ignored so callers get the same behavior regardless
    of whether their inputs came from sparse target weights or a dense panel.
    """

    if trade_weights is None or trade_weights.empty:
        return TurnoverBreakdown(0.0, 0.0, 0.0, 0.0, 0.0, is_initial=is_initial)

    clean = pd.to_numeric(trade_weights, errors="coerce")
    clean = clean.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return TurnoverBreakdown(0.0, 0.0, 0.0, 0.0, 0.0, is_initial=is_initial)

    buy_weight = float(clean.clip(lower=0.0).sum())
    sell_weight = float((-clean.clip(upper=0.0)).sum())
    gross_traded_weight = buy_weight + sell_weight
    half_l1_turnover = 0.5 * gross_traded_weight
    one_way_turnover = gross_traded_weight if is_initial else half_l1_turnover
    return TurnoverBreakdown(
        buy_weight=buy_weight,
        sell_weight=sell_weight,
        gross_traded_weight=gross_traded_weight,
        half_l1_turnover=half_l1_turnover,
        one_way_turnover=one_way_turnover,
        is_initial=is_initial,
    )


def name_turnover(
    previous_holdings: Iterable[str] | None,
    current_holdings: Iterable[str] | None,
    *,
    initial_value: float = 1.0,
) -> float:
    """Return holding-name replacement, separate from weight turnover.

    The denominator is the current holding count, matching the historical
    Top-K estimate. An initial portfolio reports ``initial_value`` when it is
    non-empty; callers that do not want an initial observation can simply skip
    it.
    """

    current = {str(value) for value in current_holdings or () if pd.notna(value)}
    if previous_holdings is None:
        return float(initial_value) if current else 0.0

    previous = {str(value) for value in previous_holdings if pd.notna(value)}
    if not current:
        return 1.0 if previous else 0.0
    overlap = len(previous & current)
    return float(1.0 - overlap / len(current))


def annualize_turnover(turnover: float, *, periods_per_year: float = 252.0) -> float:
    """Linearly annualize a per-period turnover observation or mean."""

    value = float(turnover)
    periods = float(periods_per_year)
    if not np.isfinite(value) or not np.isfinite(periods) or periods < 0:
        return np.nan
    return value * periods
