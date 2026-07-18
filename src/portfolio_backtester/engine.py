from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Literal, cast

import pandas as pd

from .api import backtest_topk as backtest_topk
from .execution import CostModel, SelectionConstraints, SlippageModel
from .leg_helpers import (
    _build_backtest_leg_result,
    _build_target_weights_and_exit,
    _compute_trade_summary as _compute_trade_summary,
    _next_position_state,
)
from .period_turnover import (
    period_result_from_leg,
    period_result_from_legs,
    period_turnover_fields,
)
from .periods import resolve_backtest_period_plan
from .portfolio_selection import select_holdings
from .portfolio_weights import normalize_position_weights, validate_positive_name_invariant
from .selection_controls import MaxNewNamesShortfallPolicy, controlled_selection_day
from .topk_context import (
    _BacktestPeriodEvaluation,
    _BacktestResultAccumulator,
    _BacktestRunContext,
    _BacktestTopKConfig,
    _build_backtest_return_bundle,
    _prepare_backtest_run_context,
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
    cost_model: CostModel
    slippage_model: SlippageModel
    exit_policy: object
    date_to_idx: dict[pd.Timestamp, int]


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
    cash_control_enabled = (
        context.selection_min_score is not None or context.max_new_names_per_rebalance is not None
    )
    if count <= 0 and not cash_control_enabled:
        return None
    if count > 0:
        holdings, entry_prices = select_holdings(
            context.day,
            context.entry_date,
            count,
            context.pred_col,
            ascending=ascending,
            price_table=context.entry_price_table,
            tradable_table=context.tradable_table,
            amount_table=context.amount_tables.get(context.selection_constraints.amount_col),
            constraints=context.selection_constraints,
            prev_holdings=previous.holdings,
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
        )
    else:
        holdings, entry_prices = [], pd.Series(dtype=float)
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
        planned_exit_idx=context.planned_exit_idx,
        resolve_exit_prices=_backtest_exit_price_resolver(context),
    )
    if target is None:
        return None
    weights, exit_prices = target.requested_weights, target.exit_prices
    if cash_control_enabled:
        entry_prices = context.entry_price_table.loc[context.entry_date].reindex(exit_prices.index)
    else:
        entry_prices = entry_prices.reindex(exit_prices.index)
    weights = normalize_position_weights(weights.reindex(exit_prices.index))
    weights = validate_positive_name_invariant(weights, context.max_positive_names)
    holdings = cast(list[str], list(weights.index))
    if not holdings and not cash_control_enabled:
        return None
    entry_prices = entry_prices.reindex(holdings)
    exit_prices = exit_prices.reindex(holdings)
    return _build_backtest_leg_result(
        holdings=holdings,
        target_weights=target.target_weights,
        weights=weights,
        entry_prices=entry_prices,
        exit_prices=exit_prices,
        period_exit_idx=target.exit_idx,
        entry_idx=context.entry_idx,
        entry_date=context.entry_date,
        trade_dates=context.trade_dates,
        entry_price_table=context.entry_price_table,
        side=side,
        previous=previous,
        cost_model=context.cost_model,
        slippage_model=context.slippage_model,
        amount_tables=context.amount_tables,
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


def _append_backtest_period_result(
    *,
    period_result: BacktestPeriodResult,
    reb_date: pd.Timestamp,
    entry_idx: int,
    planned_exit_idx: int,
    entry_date: pd.Timestamp,
    planned_exit_date: pd.Timestamp,
    net_returns: list[float],
    gross_returns: list[float],
    turnovers: list[float],
    costs: list[float],
    fee_costs: list[float],
    slippage_costs: list[float],
    period_info: list[dict],
) -> None:
    gross_returns.append(period_result.gross)
    net_returns.append(period_result.net)
    turnovers.append(period_result.turnover)
    costs.append(period_result.total_cost)
    fee_costs.append(period_result.fee_cost)
    slippage_costs.append(period_result.slippage_cost)
    period_info.append(
        {
            "rebalance_date": reb_date,
            "entry_idx": entry_idx,
            "planned_exit_idx": planned_exit_idx,
            "exit_idx": period_result.exit_idx,
            "entry_date": entry_date,
            "planned_exit_date": planned_exit_date,
            "exit_date": period_result.exit_date,
            "exit_delay_steps": int(period_result.exit_idx - planned_exit_idx),
            **period_turnover_fields(period_result),
        }
    )


def _configured_leg_context(
    day: pd.DataFrame,
    *,
    entry_date: pd.Timestamp,
    entry_idx: int,
    planned_exit_idx: int,
    config: _BacktestTopKConfig,
    run_context: _BacktestRunContext,
) -> _BacktestLegContext:
    pricing_context = run_context.pricing_context
    execution_context = run_context.execution_context
    return _BacktestLegContext(
        day=day,
        entry_date=entry_date,
        entry_idx=entry_idx,
        planned_exit_idx=planned_exit_idx,
        trade_dates=pricing_context.trade_dates,
        pred_col=config.pred_col,
        weighting_mode=run_context.weighting_mode,
        entry_price_table=pricing_context.entry_price_table,
        exit_price_table=pricing_context.exit_price_table,
        tradable_table=pricing_context.tradable_table,
        amount_tables=pricing_context.amount_tables,
        selection_constraints=execution_context.selection_constraints,
        buffer_exit=config.buffer_exit,
        buffer_entry=config.buffer_entry,
        group_col=config.group_col,
        max_names_per_group=config.max_names_per_group,
        weighting_liquidity_col=config.weighting_liquidity_col,
        selection_tiebreak_col=config.selection_tiebreak_col,
        selection_score_bucket_size=config.selection_score_bucket_size,
        selection_score_margin=config.selection_score_margin,
        selection_score_margin_col=config.selection_score_margin_col,
        selection_score_margin_rank_limit=config.selection_score_margin_rank_limit,
        selection_min_score=config.selection_min_score,
        max_new_names_per_rebalance=config.max_new_names_per_rebalance,
        max_new_names_shortfall_policy=config.max_new_names_shortfall_policy,
        max_positive_names=config.max_positive_names,
        cost_model=execution_context.cost_model,
        slippage_model=execution_context.slippage_model,
        exit_policy=execution_context.exit_policy,
        date_to_idx=pricing_context.date_to_idx,
    )


def _evaluate_configured_long_only_period(
    day: pd.DataFrame,
    *,
    entry_date: pd.Timestamp,
    entry_idx: int,
    planned_exit_idx: int,
    count: int,
    long_state: BacktestPositionState,
    config: _BacktestTopKConfig,
    run_context: _BacktestRunContext,
) -> tuple[BacktestPeriodResult, BacktestPositionState] | None:
    context = _configured_leg_context(
        day,
        entry_date=entry_date,
        entry_idx=entry_idx,
        planned_exit_idx=planned_exit_idx,
        config=config,
        run_context=run_context,
    )
    return _evaluate_long_only_period(
        context,
        count=count,
        long_state=long_state,
        rank_offset=config.rank_offset,
        max_turnover_per_rebalance=config.max_turnover_per_rebalance,
    )


def _evaluate_configured_long_short_period(
    day: pd.DataFrame,
    *,
    entry_date: pd.Timestamp,
    entry_idx: int,
    planned_exit_idx: int,
    long_count: int,
    short_count: int,
    long_state: BacktestPositionState,
    short_state: BacktestPositionState,
    config: _BacktestTopKConfig,
    run_context: _BacktestRunContext,
) -> tuple[BacktestPeriodResult, BacktestPositionState, BacktestPositionState] | None:
    context = _configured_leg_context(
        day,
        entry_date=entry_date,
        entry_idx=entry_idx,
        planned_exit_idx=planned_exit_idx,
        config=config,
        run_context=run_context,
    )
    return _evaluate_long_short_period(
        context,
        long_count=long_count,
        short_count=short_count,
        long_state=long_state,
        short_state=short_state,
        rank_offset=config.rank_offset,
        max_turnover_per_rebalance=config.max_turnover_per_rebalance,
    )


def _evaluate_backtest_rebalance_period(
    *,
    rebalance_index: int,
    reb_date: pd.Timestamp,
    accumulator: _BacktestResultAccumulator,
    config: _BacktestTopKConfig,
    run_context: _BacktestRunContext,
) -> _BacktestPeriodEvaluation | None:
    reb_date = cast(pd.Timestamp, pd.Timestamp(reb_date)).normalize()
    pricing_context = run_context.pricing_context
    execution_context = run_context.execution_context
    period_plan = resolve_backtest_period_plan(
        rebalance_dates=config.rebalance_dates,
        rebalance_index=rebalance_index,
        rebalance_date=reb_date,
        exit_mode=config.exit_mode,
        exit_horizon_days=config.exit_horizon_days,
        shift_days=config.shift_days,
        prev_exit_idx=accumulator.prev_exit_idx,
        trade_dates=pricing_context.trade_dates,
        date_to_idx=pricing_context.date_to_idx,
        execution_calendar=execution_context.calendar,
        execution_open_dates=execution_context.open_dates,
        execution_closed_dates=execution_context.closed_dates,
    )
    if period_plan is None:
        return None

    day = pricing_context.day_groups.get(reb_date)
    if day is None or day.empty:
        return None
    day = _apply_liquidity_floor(
        day,
        liquidity_floor_col=config.liquidity_floor_col,
        liquidity_floor_quantile=config.liquidity_floor_quantile,
    )
    if day.empty:
        return None

    k = min(config.top_k, max(0, len(day) - int(config.rank_offset)))
    if k <= 0:
        return None

    if config.long_only:
        period_result, long_state = _evaluate_configured_long_only_period(
            day,
            entry_date=period_plan.entry_date,
            entry_idx=period_plan.entry_idx,
            planned_exit_idx=period_plan.planned_exit_idx,
            count=k,
            long_state=accumulator.long_state,
            config=config,
            run_context=run_context,
        ) or (None, accumulator.long_state)
        if period_result is None:
            return None
        short_state = accumulator.short_state
    else:
        short_k_final = config.short_k if config.short_k is not None else k
        short_capacity = len(day) - int(config.rank_offset)
        if config.selection_min_score is None and config.max_new_names_per_rebalance is None:
            short_capacity -= k
        short_k_final = min(int(short_k_final), short_capacity)
        if short_k_final <= 0:
            return None
        long_short_result = _evaluate_configured_long_short_period(
            day,
            entry_date=period_plan.entry_date,
            entry_idx=period_plan.entry_idx,
            planned_exit_idx=period_plan.planned_exit_idx,
            long_count=k,
            short_count=short_k_final,
            long_state=accumulator.long_state,
            short_state=accumulator.short_state,
            config=config,
            run_context=run_context,
        )
        if long_short_result is None:
            return None
        period_result, long_state, short_state = long_short_result

    return _BacktestPeriodEvaluation(
        period_result=period_result,
        reb_date=reb_date,
        entry_idx=period_plan.entry_idx,
        planned_exit_idx=period_plan.planned_exit_idx,
        entry_date=period_plan.entry_date,
        planned_exit_date=period_plan.planned_exit_date,
        long_state=long_state,
        short_state=short_state,
    )


def _run_backtest_periods(
    *,
    config: _BacktestTopKConfig,
    run_context: _BacktestRunContext,
) -> _BacktestResultAccumulator:
    accumulator = _BacktestResultAccumulator()
    for i, reb_date in enumerate(config.rebalance_dates):
        evaluation = _evaluate_backtest_rebalance_period(
            rebalance_index=i,
            reb_date=reb_date,
            accumulator=accumulator,
            config=config,
            run_context=run_context,
        )
        if evaluation is None:
            continue
        _append_backtest_period_result(
            period_result=evaluation.period_result,
            reb_date=evaluation.reb_date,
            entry_idx=evaluation.entry_idx,
            planned_exit_idx=evaluation.planned_exit_idx,
            entry_date=evaluation.entry_date,
            planned_exit_date=evaluation.planned_exit_date,
            net_returns=accumulator.net_returns,
            gross_returns=accumulator.gross_returns,
            turnovers=accumulator.turnovers,
            costs=accumulator.costs,
            fee_costs=accumulator.fee_costs,
            slippage_costs=accumulator.slippage_costs,
            period_info=accumulator.period_info,
        )
        accumulator.long_state = evaluation.long_state
        accumulator.short_state = evaluation.short_state
        accumulator.prev_exit_idx = evaluation.period_result.exit_idx
    return accumulator


def _run_backtest_config(
    data: pd.DataFrame,
    *,
    config: _BacktestTopKConfig,
):
    run_context = _prepare_backtest_run_context(data, config=config)
    if run_context is None:
        return None
    accumulator = _run_backtest_periods(config=config, run_context=run_context)
    if not accumulator.net_returns:
        return None
    return _build_backtest_return_bundle(
        accumulator=accumulator,
        config=config,
        weighting_mode=run_context.weighting_mode,
    )
