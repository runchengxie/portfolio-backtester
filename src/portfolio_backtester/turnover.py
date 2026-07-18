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


@dataclass(frozen=True)
class RebalanceTurnoverReport:
    """Auditable turnover layers for one rebalance.

    Target weights compare the new signal target with the previous period's
    requested weights before current-period price drift. Pre-trade demand
    compares the portfolio after drift with the weights the backtest asks to
    hold. Executed fields are optional because a score backtest does not
    observe fills; leaving them as
    ``None`` is materially different from claiming zero execution.

    Weight and turnover values are fractions of portfolio net asset value, so
    ``1.0`` means 100% of NAV. ``executed_cost`` uses the same starting-NAV
    return-drag unit rather than currency notional.
    """

    target_name_turnover: float
    target_weight_full_l1: float
    target_weight_half_l1: float
    pretrade_demand_buy: float
    pretrade_demand_sell: float
    pretrade_demand_full_l1: float
    pretrade_demand_half_l1: float
    is_initial_build: bool
    target_entered_names: tuple[str, ...] = ()
    target_exited_names: tuple[str, ...] = ()
    target_overlap_names: tuple[str, ...] = ()
    executed_buy: float | None = None
    executed_sell: float | None = None
    executed_gross: float | None = None
    executed_full_l1: float | None = None
    executed_half_l1: float | None = None
    executed_cost: float | None = None

    @property
    def execution_data_available(self) -> bool:
        return self.executed_full_l1 is not None

    @property
    def target_entered_count(self) -> int:
        return len(self.target_entered_names)

    @property
    def target_exited_count(self) -> int:
        return len(self.target_exited_names)

    @property
    def target_overlap_count(self) -> int:
        return len(self.target_overlap_names)

    def to_dict(self) -> dict[str, float | bool | int | tuple[str, ...] | None]:
        return {
            "target_name_turnover": self.target_name_turnover,
            "target_entered_names": self.target_entered_names,
            "target_exited_names": self.target_exited_names,
            "target_overlap_names": self.target_overlap_names,
            "target_entered_count": self.target_entered_count,
            "target_exited_count": self.target_exited_count,
            "target_overlap_count": self.target_overlap_count,
            "target_weight_full_l1": self.target_weight_full_l1,
            "target_weight_half_l1": self.target_weight_half_l1,
            "pretrade_demand_buy": self.pretrade_demand_buy,
            "pretrade_demand_sell": self.pretrade_demand_sell,
            "pretrade_demand_full_l1": self.pretrade_demand_full_l1,
            "pretrade_demand_half_l1": self.pretrade_demand_half_l1,
            "executed_buy": self.executed_buy,
            "executed_sell": self.executed_sell,
            "executed_gross": self.executed_gross,
            "executed_full_l1": self.executed_full_l1,
            "executed_half_l1": self.executed_half_l1,
            "executed_cost": self.executed_cost,
            "execution_data_available": self.execution_data_available,
            "is_initial_build": self.is_initial_build,
        }


def _aligned_weight_delta(
    previous: pd.Series | None,
    current: pd.Series,
) -> pd.Series:
    current_clean = pd.to_numeric(current, errors="coerce").replace([np.inf, -np.inf], np.nan)
    current_clean = current_clean.dropna()
    if previous is None:
        return current_clean
    previous_clean = pd.to_numeric(previous, errors="coerce").replace([np.inf, -np.inf], np.nan)
    previous_clean = previous_clean.dropna()
    symbols = previous_clean.index.union(current_clean.index)
    return current_clean.reindex(symbols).fillna(0.0) - previous_clean.reindex(symbols).fillna(0.0)


def _normalized_name_set(values: Iterable[str] | None) -> set[str]:
    return {str(value) for value in (() if values is None else values) if pd.notna(value)}


def _target_name_changes(
    previous_holdings: Iterable[str] | None,
    target_holdings: Iterable[str],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    previous = _normalized_name_set(previous_holdings)
    target = _normalized_name_set(target_holdings)
    return (
        tuple(sorted(target - previous)),
        tuple(sorted(previous - target)),
        tuple(sorted(previous & target)),
    )


def build_rebalance_turnover_report(
    *,
    previous_holdings: Iterable[str] | None,
    target_holdings: Iterable[str],
    previous_target_weights: pd.Series | None,
    target_weights: pd.Series,
    pretrade_trade_weights: pd.Series,
    executed_trade_weights: pd.Series | None = None,
    executed_cost: float | None = None,
) -> RebalanceTurnoverReport:
    """Build explicit target, demand, and optional execution turnover layers."""

    if executed_cost is not None and executed_trade_weights is None:
        raise ValueError("executed_cost requires executed_trade_weights.")
    is_initial = previous_target_weights is None
    target_delta = turnover_from_trade_weights(
        _aligned_weight_delta(previous_target_weights, target_weights),
        is_initial=is_initial,
    )
    demand = turnover_from_trade_weights(pretrade_trade_weights, is_initial=is_initial)
    executed = (
        turnover_from_trade_weights(executed_trade_weights, is_initial=is_initial)
        if executed_trade_weights is not None
        else None
    )
    entered, exited, overlap = _target_name_changes(previous_holdings, target_holdings)
    return RebalanceTurnoverReport(
        target_name_turnover=name_turnover(previous_holdings, target_holdings),
        target_entered_names=entered,
        target_exited_names=exited,
        target_overlap_names=overlap,
        target_weight_full_l1=target_delta.gross_traded_weight,
        target_weight_half_l1=target_delta.half_l1_turnover,
        pretrade_demand_buy=demand.buy_weight,
        pretrade_demand_sell=demand.sell_weight,
        pretrade_demand_full_l1=demand.gross_traded_weight,
        pretrade_demand_half_l1=demand.half_l1_turnover,
        executed_buy=executed.buy_weight if executed is not None else None,
        executed_sell=executed.sell_weight if executed is not None else None,
        executed_gross=executed.gross_traded_weight if executed is not None else None,
        executed_full_l1=executed.gross_traded_weight if executed is not None else None,
        executed_half_l1=executed.half_l1_turnover if executed is not None else None,
        executed_cost=float(executed_cost) if executed_cost is not None else None,
        is_initial_build=is_initial,
    )


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

    current = _normalized_name_set(current_holdings)
    if previous_holdings is None:
        return float(initial_value) if current else 0.0

    previous = _normalized_name_set(previous_holdings)
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
