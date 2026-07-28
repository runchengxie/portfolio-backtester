"""Backtest engine: leg context, holding selection, and leg evaluation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Literal, cast

import pandas as pd

from .execution import CostModel, SelectionConstraints, SlippageModel
from .holding_selection import filter_entry_eligible_symbols
from .leg_helpers import (
    _build_backtest_leg_result,
    _build_target_weights_and_exit,
    _next_position_state,
)
from .period_turnover import period_result_from_leg, period_result_from_legs
from .portfolio_selection import select_holdings
from .portfolio_weights import clean_position_weights, validate_positive_name_invariant
from .selection_controls import (
    MaxNewNamesShortfallPolicy,
    SelectionPricePolicy,
    TargetWeightPolicy,
    controlled_selection_day,
    entry_amount_values,
    entry_tradable_flags,
)
from .types import BacktestLegResult, BacktestPeriodResult, BacktestPositionState


@dataclass(frozen=True)
class _BacktestLegContext:
    day: pd.DataFrame
    entry_date: pd.Timestamp
    entry_idx: int
    planned_exit_idx: int
    trade_dates: list[pd.Timestamp]
    pred_col: str
    weighting_mode: str
    entry_price_table: pd.DataFrame
    exit_price_table: pd.DataFrame
    tradable_table: pd.DataFrame | None
    amount_tables: dict[str, pd.DataFrame]
    selection_constraints: SelectionConstraints
    buffer_exit: int
    buffer_entry: int
    group_col: str | None
    max_names_per_group: int | None
    weighting_liquidity_col: str
    selection_tiebreak_col: str | None
    selection_score_bucket_size: float | None
    selection_score_margin: float | None
    selection_score_margin_col: str | None
    selection_score_margin_rank_limit: int | None
    selection_min_score: float | None
    max_new_names_per_rebalance: int | None
    max_new_names_shortfall_policy: MaxNewNamesShortfallPolicy
    max_positive_names: int | None
    entry_rank_cutoff: int | None
    selection_price_policy: SelectionPricePolicy
    target_weight_policy: TargetWeightPolicy
    target_slot_count: int
    cost_model: CostModel
    slippage_model: SlippageModel
    exit_policy: object
    date_to_idx: dict[pd.Timestamp, int]


@dataclass(frozen=True)
class _ExecutableLeg:
    holdings: list[str]
    weights: pd.Series
    entry_prices: pd.Series
    exit_prices: pd.Series
    exit_idx: int


def _cash_target_allowed(context: _BacktestLegContext) -> bool:
    return bool(
        context.selection_min_score is not None
        or context.max_new_names_per_rebalance is not None
        or context.entry_rank_cutoff is not None
        or context.target_weight_policy == "fixed_slot"
        or context.selection_price_policy == "target_first"
    )


def _select_target_holdings(
    context: _BacktestLegContext,
    *,
    count: int,
    ascending: bool,
    previous: BacktestPositionState,
    rank_offset: int,
) -> list[str]:
    if count <= 0:
        return []
    previous_holdings = (
        previous.target_holdings
        if context.selection_price_policy == "target_first" and previous.target_holdings is not None
        else previous.holdings
    )
    holdings, _ = select_holdings(
        context.day,
        context.entry_date,
        count,
        context.pred_col,
        ascending=ascending,
        price_table=context.entry_price_table,
        tradable_table=context.tradable_table,
        amount_table=context.amount_tables.get(context.selection_constraints.amount_col),
        constraints=context.selection_constraints,
        prev_holdings=previous_holdings,
        buffer_exit=context.buffer_exit,
        buffer_entry=context.buffer_entry,
        rank_offset=rank_offset,
        group_col=context.group_col,
        max_names_per_group=context.max_names_per_group,
        selection_tiebreak_col=context.selection_tiebreak_col,
        selection_score_bucket_size=context.selection_score_bucket_size,
        selection_score_margin=context.selection_score_margin,
        selection_score_margin_col=context.selection_score_margin_col,
        selection_score_margin_rank_limit=context.selection_score_margin_rank_limit,
        selection_min_score=context.selection_min_score,
        max_new_names_per_rebalance=context.max_new_names_per_rebalance,
        max_new_names_shortfall_policy=context.max_new_names_shortfall_policy,
        entry_rank_cutoff=context.entry_rank_cutoff,
        selection_price_policy=context.selection_price_policy,
    )
    return holdings


def _resolve_executable_leg(
    context: _BacktestLegContext,
    requested_weights: pd.Series,
    *,
    preserve_gross_exposure: bool,
) -> _ExecutableLeg | None:
    all_entry_prices = context.entry_price_table.loc[context.entry_date]
    amount_values = entry_amount_values(
        constraints=context.selection_constraints,
        amount_table=context.amount_tables.get(context.selection_constraints.amount_col),
        lookup_date=context.entry_date,
    )
    executable_holdings = filter_entry_eligible_symbols(
        [str(symbol) for symbol in requested_weights.index],
        entry_prices=all_entry_prices,
        amount_values=amount_values,
        tradable_flags=entry_tradable_flags(context.tradable_table, context.entry_date),
        constraints=context.selection_constraints,
    )
    if executable_holdings:
        exit_prices, exit_idx = _backtest_exit_price_resolver(context)(
            executable_holdings,
            context.planned_exit_idx,
        )
        if exit_prices.empty:
            return None
    else:
        exit_prices = pd.Series(dtype=float)
        exit_idx = context.planned_exit_idx
    weights = clean_position_weights(
        requested_weights.reindex(exit_prices.index).dropna(),
        preserve_gross_exposure=preserve_gross_exposure,
    )
    weights = validate_positive_name_invariant(weights, context.max_positive_names)
    holdings = cast(list[str], list(weights.index))
    return _ExecutableLeg(
        holdings=holdings,
        weights=weights,
        entry_prices=all_entry_prices.reindex(holdings),
        exit_prices=exit_prices.reindex(holdings),
        exit_idx=exit_idx,
    )


def _evaluate_backtest_leg(
    context: _BacktestLegContext,
    *,
    side: Literal["long", "short"],
    count: int,
    ascending: bool,
    previous: BacktestPositionState,
    rank_offset: int,
    max_turnover_per_rebalance: float | None,
) -> BacktestLegResult | None:
    preserve_gross_exposure = (
        context.target_weight_policy == "fixed_slot"
        or context.selection_price_policy == "target_first"
    )
    cash_control_enabled = _cash_target_allowed(context)
    if count <= 0 and not cash_control_enabled:
        return None
    holdings = _select_target_holdings(
        context,
        count=count,
        ascending=ascending,
        previous=previous,
        rank_offset=rank_offset,
    )
    if not holdings and not cash_control_enabled:
        return None
    weighting_day = controlled_selection_day(
        context.day,
        context.pred_col,
        ascending=ascending,
        selection_tiebreak_col=context.selection_tiebreak_col,
        selection_score_bucket_size=context.selection_score_bucket_size,
        selection_min_score=context.selection_min_score,
        max_new_names_per_rebalance=context.max_new_names_per_rebalance,
    )
    target = _build_target_weights_and_exit(
        day=weighting_day,
        holdings=holdings,
        pred_col=context.pred_col,
        side=side,
        weighting_mode=context.weighting_mode,
        weighting_liquidity_col=context.weighting_liquidity_col,
        previous=previous,
        max_turnover_per_rebalance=max_turnover_per_rebalance,
        selection_min_score=context.selection_min_score,
        target_weight_policy=context.target_weight_policy,
        target_slot_count=context.target_slot_count,
        preserve_gross_exposure=preserve_gross_exposure,
    )
    executable = _resolve_executable_leg(
        context,
        target.requested_weights,
        preserve_gross_exposure=preserve_gross_exposure,
    )
    if executable is None:
        return None
    if not executable.holdings and not cash_control_enabled:
        return None
    return _build_backtest_leg_result(
        holdings=executable.holdings,
        target_weights=target.target_weights,
        weights=executable.weights,
        entry_prices=executable.entry_prices,
        exit_prices=executable.exit_prices,
        period_exit_idx=executable.exit_idx,
        entry_idx=context.entry_idx,
        entry_date=context.entry_date,
        trade_dates=context.trade_dates,
        entry_price_table=context.entry_price_table,
        side=side,
        previous=previous,
        cost_model=context.cost_model,
        slippage_model=context.slippage_model,
        amount_tables=context.amount_tables,
        preserve_gross_exposure=preserve_gross_exposure,
    )


def _apply_liquidity_floor(
    day: pd.DataFrame,
    *,
    liquidity_floor_col: str | None,
    liquidity_floor_quantile: float | None,
) -> pd.DataFrame:
    if not liquidity_floor_col or liquidity_floor_quantile is None:
        return day
    if liquidity_floor_col not in day.columns:
        raise ValueError(f"Backtest liquidity floor column not found: {liquidity_floor_col}")
    floor_q = float(liquidity_floor_quantile)
    if floor_q <= 0:
        return day
    liquidity = cast(
        pd.Series,
        pd.to_numeric(cast(pd.Series, day[liquidity_floor_col]), errors="coerce"),
    )
    if liquidity.notna().sum() <= 1:
        return day
    cutoff = liquidity.quantile(floor_q)
    return cast(pd.DataFrame, day.loc[liquidity.isna() | (liquidity >= cutoff)].copy())


def _resolve_exit_prices_for_policy(
    *,
    exit_policy,
    holdings: list[str],
    planned_exit_idx: int,
    exit_price_table: pd.DataFrame,
    tradable_table: pd.DataFrame | None,
    trade_dates: list[pd.Timestamp],
    date_to_idx: dict[pd.Timestamp, int],
) -> tuple[pd.Series, int]:
    return exit_policy.resolve_exit_prices(
        holdings,
        planned_exit_idx,
        price_table=exit_price_table,
        tradable_table=tradable_table,
        trade_dates=trade_dates,
        date_to_idx=date_to_idx,
    )


def _evaluate_long_only_period(
    context: _BacktestLegContext,
    *,
    count: int,
    long_state: BacktestPositionState,
    rank_offset: int,
    max_turnover_per_rebalance: float | None,
) -> tuple[BacktestPeriodResult, BacktestPositionState] | None:
    long_leg = _evaluate_paired_backtest_leg(
        context,
        side="long",
        count=count,
        ascending=False,
        previous=long_state,
        rank_offset=rank_offset,
        max_turnover_per_rebalance=max_turnover_per_rebalance,
    )
    if long_leg is None:
        return None
    result = period_result_from_leg(long_leg)
    next_state = _next_position_state(long_leg, entry_date=context.entry_date)
    return result, next_state


def _backtest_exit_price_resolver(
    context: _BacktestLegContext,
) -> Callable[[list[str], int], tuple[pd.Series, int]]:
    def resolve_exit_prices(holdings: list[str], planned_exit: int) -> tuple[pd.Series, int]:
        return _resolve_exit_prices_for_policy(
            exit_policy=context.exit_policy,
            holdings=holdings,
            planned_exit_idx=planned_exit,
            exit_price_table=context.exit_price_table,
            tradable_table=context.tradable_table,
            trade_dates=context.trade_dates,
            date_to_idx=context.date_to_idx,
        )

    return resolve_exit_prices


def _evaluate_paired_backtest_leg(
    context: _BacktestLegContext,
    *,
    side: Literal["long", "short"],
    count: int,
    ascending: bool,
    previous: BacktestPositionState,
    rank_offset: int,
    max_turnover_per_rebalance: float | None,
) -> BacktestLegResult | None:
    return _evaluate_backtest_leg(
        context,
        side=side,
        count=count,
        ascending=ascending,
        previous=previous,
        rank_offset=rank_offset,
        max_turnover_per_rebalance=max_turnover_per_rebalance,
    )


def _evaluate_long_short_period(
    context: _BacktestLegContext,
    *,
    long_count: int,
    short_count: int,
    long_state: BacktestPositionState,
    short_state: BacktestPositionState,
    rank_offset: int,
    max_turnover_per_rebalance: float | None,
) -> tuple[BacktestPeriodResult, BacktestPositionState, BacktestPositionState] | None:
    long_leg = _evaluate_paired_backtest_leg(
        context,
        side="long",
        count=long_count,
        ascending=False,
        previous=long_state,
        rank_offset=rank_offset,
        max_turnover_per_rebalance=max_turnover_per_rebalance,
    )
    short_context = context
    short_count_final = short_count
    if (
        context.selection_min_score is not None or context.max_new_names_per_rebalance is not None
    ) and long_leg is not None:
        short_day = context.day.loc[~context.day["symbol"].isin(long_leg.holdings)].copy()
        short_context = replace(context, day=short_day)
        short_count_final = min(short_count, len(short_day))
    short_leg = _evaluate_paired_backtest_leg(
        short_context,
        side="short",
        count=short_count_final,
        ascending=True,
        previous=short_state,
        rank_offset=0,
        max_turnover_per_rebalance=None,
    )
    if long_leg is None or short_leg is None:
        return None
    result = period_result_from_legs(long_leg, short_leg, trade_dates=context.trade_dates)
    next_long = _next_position_state(long_leg, entry_date=context.entry_date)
    next_short = _next_position_state(short_leg, entry_date=context.entry_date)
    return result, next_long, next_short
