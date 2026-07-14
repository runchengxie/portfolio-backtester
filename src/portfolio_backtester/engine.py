from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, cast

import pandas as pd

from .api import backtest_topk as backtest_topk
from .execution import CostModel, SelectionConstraints, SlippageModel
from .leg_helpers import (
    _build_backtest_leg_result,
    _compute_trade_summary as _compute_trade_summary,
)
from .periods import resolve_backtest_period_plan
from .portfolio_selection import select_holdings
from .portfolio_weights import (
    build_position_weights,
    limit_weight_turnover,
    normalize_position_weights,
)
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
class _PairedLegContext:
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
    selection_score_margin_rank_limit: int | None
    cost_model: CostModel
    slippage_model: SlippageModel
    exit_policy: object
    date_to_idx: dict[pd.Timestamp, int]


def _evaluate_backtest_leg(
    *,
    day: pd.DataFrame,
    entry_date: pd.Timestamp,
    entry_idx: int,
    planned_exit_idx: int,
    trade_dates: list[pd.Timestamp],
    pred_col: str,
    side: Literal["long", "short"],
    count: int,
    ascending: bool,
    weighting_mode: str,
    entry_price_table: pd.DataFrame,
    tradable_table: pd.DataFrame | None,
    amount_tables: dict[str, pd.DataFrame],
    selection_constraints: SelectionConstraints,
    previous: BacktestPositionState,
    buffer_exit: int,
    buffer_entry: int,
    rank_offset: int,
    group_col: str | None,
    max_names_per_group: int | None,
    weighting_liquidity_col: str,
    max_turnover_per_rebalance: float | None,
    selection_tiebreak_col: str | None,
    selection_score_bucket_size: float | None,
    selection_score_margin: float | None,
    selection_score_margin_rank_limit: int | None,
    cost_model: CostModel,
    slippage_model: SlippageModel,
    resolve_exit_prices,
) -> BacktestLegResult | None:
    if count <= 0:
        return None

    holdings, entry_prices = select_holdings(
        day,
        entry_date,
        count,
        pred_col,
        ascending=ascending,
        price_table=entry_price_table,
        tradable_table=tradable_table,
        amount_table=amount_tables.get(selection_constraints.amount_col),
        constraints=selection_constraints,
        prev_holdings=previous.holdings,
        buffer_exit=buffer_exit,
        buffer_entry=buffer_entry,
        rank_offset=rank_offset,
        group_col=group_col,
        max_names_per_group=max_names_per_group,
        selection_tiebreak_col=selection_tiebreak_col,
        selection_score_bucket_size=selection_score_bucket_size,
        selection_score_margin=selection_score_margin,
        selection_score_margin_rank_limit=selection_score_margin_rank_limit,
    )
    if not holdings:
        return None

    weights = build_position_weights(
        day,
        holdings,
        pred_col,
        side=side,
        weighting=weighting_mode,
        liquidity_col=weighting_liquidity_col,
    )
    weights = limit_weight_turnover(
        previous.weights,
        weights,
        max_turnover_per_rebalance,
    )
    exit_prices, period_exit_idx = resolve_exit_prices(list(weights.index), planned_exit_idx)
    if exit_prices.empty:
        return None

    entry_prices = entry_prices.reindex(exit_prices.index)
    weights = normalize_position_weights(weights.reindex(exit_prices.index))
    holdings = cast(list[str], list(weights.index))
    if not holdings:
        return None

    entry_prices = entry_prices.reindex(holdings)
    exit_prices = exit_prices.reindex(holdings)
    return _build_backtest_leg_result(
        holdings=holdings,
        weights=weights,
        entry_prices=entry_prices,
        exit_prices=exit_prices,
        period_exit_idx=period_exit_idx,
        entry_idx=entry_idx,
        entry_date=entry_date,
        trade_dates=trade_dates,
        entry_price_table=entry_price_table,
        side=side,
        previous=previous,
        cost_model=cost_model,
        slippage_model=slippage_model,
        amount_tables=amount_tables,
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
    *,
    day: pd.DataFrame,
    entry_date: pd.Timestamp,
    entry_idx: int,
    planned_exit_idx: int,
    trade_dates: list[pd.Timestamp],
    pred_col: str,
    count: int,
    weighting_mode: str,
    entry_price_table: pd.DataFrame,
    exit_price_table: pd.DataFrame,
    tradable_table: pd.DataFrame | None,
    amount_tables: dict[str, pd.DataFrame],
    selection_constraints: SelectionConstraints,
    long_state: BacktestPositionState,
    buffer_exit: int,
    buffer_entry: int,
    rank_offset: int,
    group_col: str | None,
    max_names_per_group: int | None,
    weighting_liquidity_col: str,
    max_turnover_per_rebalance: float | None,
    selection_tiebreak_col: str | None,
    selection_score_bucket_size: float | None,
    selection_score_margin: float | None,
    selection_score_margin_rank_limit: int | None,
    cost_model: CostModel,
    slippage_model: SlippageModel,
    exit_policy,
    date_to_idx: dict[pd.Timestamp, int],
) -> tuple[BacktestPeriodResult, BacktestPositionState] | None:
    def resolve_exit_prices(holdings: list[str], planned_exit: int) -> tuple[pd.Series, int]:
        return _resolve_exit_prices_for_policy(
            exit_policy=exit_policy,
            holdings=holdings,
            planned_exit_idx=planned_exit,
            exit_price_table=exit_price_table,
            tradable_table=tradable_table,
            trade_dates=trade_dates,
            date_to_idx=date_to_idx,
        )

    long_leg = _evaluate_backtest_leg(
        day=day,
        entry_date=entry_date,
        entry_idx=entry_idx,
        planned_exit_idx=planned_exit_idx,
        trade_dates=trade_dates,
        pred_col=pred_col,
        side="long",
        count=count,
        ascending=False,
        weighting_mode=weighting_mode,
        entry_price_table=entry_price_table,
        tradable_table=tradable_table,
        amount_tables=amount_tables,
        selection_constraints=selection_constraints,
        previous=long_state,
        buffer_exit=buffer_exit,
        buffer_entry=buffer_entry,
        rank_offset=rank_offset,
        group_col=group_col,
        max_names_per_group=max_names_per_group,
        weighting_liquidity_col=weighting_liquidity_col,
        max_turnover_per_rebalance=max_turnover_per_rebalance,
        selection_tiebreak_col=selection_tiebreak_col,
        selection_score_bucket_size=selection_score_bucket_size,
        selection_score_margin=selection_score_margin,
        selection_score_margin_rank_limit=selection_score_margin_rank_limit,
        cost_model=cost_model,
        slippage_model=slippage_model,
        resolve_exit_prices=resolve_exit_prices,
    )
    if long_leg is None:
        return None
    fee_cost = long_leg.fee_cost
    slippage_cost = long_leg.slippage_cost
    total_cost = fee_cost + slippage_cost
    result = BacktestPeriodResult(
        gross=long_leg.gross,
        net=long_leg.gross - total_cost,
        turnover=long_leg.turnover,
        fee_cost=fee_cost,
        slippage_cost=slippage_cost,
        total_cost=total_cost,
        exit_idx=long_leg.exit_idx,
        exit_date=long_leg.exit_date,
    )
    next_state = BacktestPositionState(
        holdings=set(long_leg.holdings),
        weights=long_leg.weights,
        entry_date=entry_date,
        entry_prices=long_leg.entry_prices,
    )
    return result, next_state


def _build_long_short_period_result(
    *,
    long_leg: BacktestLegResult,
    short_leg: BacktestLegResult,
    entry_date: pd.Timestamp,
    trade_dates: list[pd.Timestamp],
) -> tuple[BacktestPeriodResult, BacktestPositionState, BacktestPositionState]:
    exit_idx = max(long_leg.exit_idx, short_leg.exit_idx)
    fee_cost = long_leg.fee_cost + short_leg.fee_cost
    slippage_cost = long_leg.slippage_cost + short_leg.slippage_cost
    total_cost = fee_cost + slippage_cost
    gross = long_leg.gross + short_leg.gross
    result = BacktestPeriodResult(
        gross=gross,
        net=gross - total_cost,
        turnover=long_leg.turnover + short_leg.turnover,
        fee_cost=fee_cost,
        slippage_cost=slippage_cost,
        total_cost=total_cost,
        exit_idx=exit_idx,
        exit_date=trade_dates[exit_idx],
    )
    next_long = BacktestPositionState(
        holdings=set(long_leg.holdings),
        weights=long_leg.weights,
        entry_date=entry_date,
        entry_prices=long_leg.entry_prices,
    )
    next_short = BacktestPositionState(
        holdings=set(short_leg.holdings),
        weights=short_leg.weights,
        entry_date=entry_date,
        entry_prices=short_leg.entry_prices,
    )
    return result, next_long, next_short


def _paired_exit_price_resolver(
    context: _PairedLegContext,
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
    context: _PairedLegContext,
    *,
    side: Literal["long", "short"],
    count: int,
    ascending: bool,
    previous: BacktestPositionState,
    rank_offset: int,
    max_turnover_per_rebalance: float | None,
) -> BacktestLegResult | None:
    return _evaluate_backtest_leg(
        day=context.day,
        entry_date=context.entry_date,
        entry_idx=context.entry_idx,
        planned_exit_idx=context.planned_exit_idx,
        trade_dates=context.trade_dates,
        pred_col=context.pred_col,
        side=side,
        count=count,
        ascending=ascending,
        weighting_mode=context.weighting_mode,
        entry_price_table=context.entry_price_table,
        tradable_table=context.tradable_table,
        amount_tables=context.amount_tables,
        selection_constraints=context.selection_constraints,
        previous=previous,
        buffer_exit=context.buffer_exit,
        buffer_entry=context.buffer_entry,
        rank_offset=rank_offset,
        group_col=context.group_col,
        max_names_per_group=context.max_names_per_group,
        weighting_liquidity_col=context.weighting_liquidity_col,
        max_turnover_per_rebalance=max_turnover_per_rebalance,
        selection_tiebreak_col=context.selection_tiebreak_col,
        selection_score_bucket_size=context.selection_score_bucket_size,
        selection_score_margin=context.selection_score_margin,
        selection_score_margin_rank_limit=context.selection_score_margin_rank_limit,
        cost_model=context.cost_model,
        slippage_model=context.slippage_model,
        resolve_exit_prices=_paired_exit_price_resolver(context),
    )


def _evaluate_long_short_period(
    *,
    day: pd.DataFrame,
    entry_date: pd.Timestamp,
    entry_idx: int,
    planned_exit_idx: int,
    trade_dates: list[pd.Timestamp],
    pred_col: str,
    long_count: int,
    short_count: int,
    weighting_mode: str,
    entry_price_table: pd.DataFrame,
    exit_price_table: pd.DataFrame,
    tradable_table: pd.DataFrame | None,
    amount_tables: dict[str, pd.DataFrame],
    selection_constraints: SelectionConstraints,
    long_state: BacktestPositionState,
    short_state: BacktestPositionState,
    buffer_exit: int,
    buffer_entry: int,
    rank_offset: int,
    group_col: str | None,
    max_names_per_group: int | None,
    weighting_liquidity_col: str,
    max_turnover_per_rebalance: float | None,
    selection_tiebreak_col: str | None,
    selection_score_bucket_size: float | None,
    selection_score_margin: float | None,
    selection_score_margin_rank_limit: int | None,
    cost_model: CostModel,
    slippage_model: SlippageModel,
    exit_policy,
    date_to_idx: dict[pd.Timestamp, int],
) -> tuple[BacktestPeriodResult, BacktestPositionState, BacktestPositionState] | None:
    context = _PairedLegContext(
        day=day,
        entry_date=entry_date,
        entry_idx=entry_idx,
        planned_exit_idx=planned_exit_idx,
        trade_dates=trade_dates,
        pred_col=pred_col,
        weighting_mode=weighting_mode,
        entry_price_table=entry_price_table,
        exit_price_table=exit_price_table,
        tradable_table=tradable_table,
        amount_tables=amount_tables,
        selection_constraints=selection_constraints,
        buffer_exit=buffer_exit,
        buffer_entry=buffer_entry,
        group_col=group_col,
        max_names_per_group=max_names_per_group,
        weighting_liquidity_col=weighting_liquidity_col,
        selection_tiebreak_col=selection_tiebreak_col,
        selection_score_bucket_size=selection_score_bucket_size,
        selection_score_margin=selection_score_margin,
        selection_score_margin_rank_limit=selection_score_margin_rank_limit,
        cost_model=cost_model,
        slippage_model=slippage_model,
        exit_policy=exit_policy,
        date_to_idx=date_to_idx,
    )
    long_leg = _evaluate_paired_backtest_leg(
        context,
        side="long",
        count=long_count,
        ascending=False,
        previous=long_state,
        rank_offset=rank_offset,
        max_turnover_per_rebalance=max_turnover_per_rebalance,
    )
    short_leg = _evaluate_paired_backtest_leg(
        context,
        side="short",
        count=short_count,
        ascending=True,
        previous=short_state,
        rank_offset=0,
        max_turnover_per_rebalance=None,
    )
    if long_leg is None or short_leg is None:
        return None
    return _build_long_short_period_result(
        long_leg=long_leg,
        short_leg=short_leg,
        entry_date=entry_date,
        trade_dates=trade_dates,
    )


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
        }
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
    pricing_context = run_context.pricing_context
    execution_context = run_context.execution_context
    return _evaluate_long_only_period(
        day=day,
        entry_date=entry_date,
        entry_idx=entry_idx,
        planned_exit_idx=planned_exit_idx,
        trade_dates=pricing_context.trade_dates,
        pred_col=config.pred_col,
        count=count,
        weighting_mode=run_context.weighting_mode,
        entry_price_table=pricing_context.entry_price_table,
        exit_price_table=pricing_context.exit_price_table,
        tradable_table=pricing_context.tradable_table,
        amount_tables=pricing_context.amount_tables,
        selection_constraints=execution_context.selection_constraints,
        long_state=long_state,
        buffer_exit=config.buffer_exit,
        buffer_entry=config.buffer_entry,
        rank_offset=config.rank_offset,
        group_col=config.group_col,
        max_names_per_group=config.max_names_per_group,
        weighting_liquidity_col=config.weighting_liquidity_col,
        max_turnover_per_rebalance=config.max_turnover_per_rebalance,
        selection_tiebreak_col=config.selection_tiebreak_col,
        selection_score_bucket_size=config.selection_score_bucket_size,
        selection_score_margin=config.selection_score_margin,
        selection_score_margin_rank_limit=config.selection_score_margin_rank_limit,
        cost_model=execution_context.cost_model,
        slippage_model=execution_context.slippage_model,
        exit_policy=execution_context.exit_policy,
        date_to_idx=pricing_context.date_to_idx,
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
    pricing_context = run_context.pricing_context
    execution_context = run_context.execution_context
    return _evaluate_long_short_period(
        day=day,
        entry_date=entry_date,
        entry_idx=entry_idx,
        planned_exit_idx=planned_exit_idx,
        trade_dates=pricing_context.trade_dates,
        pred_col=config.pred_col,
        long_count=long_count,
        short_count=short_count,
        weighting_mode=run_context.weighting_mode,
        entry_price_table=pricing_context.entry_price_table,
        exit_price_table=pricing_context.exit_price_table,
        tradable_table=pricing_context.tradable_table,
        amount_tables=pricing_context.amount_tables,
        selection_constraints=execution_context.selection_constraints,
        long_state=long_state,
        short_state=short_state,
        buffer_exit=config.buffer_exit,
        buffer_entry=config.buffer_entry,
        rank_offset=config.rank_offset,
        group_col=config.group_col,
        max_names_per_group=config.max_names_per_group,
        weighting_liquidity_col=config.weighting_liquidity_col,
        max_turnover_per_rebalance=config.max_turnover_per_rebalance,
        selection_tiebreak_col=config.selection_tiebreak_col,
        selection_score_bucket_size=config.selection_score_bucket_size,
        selection_score_margin=config.selection_score_margin,
        selection_score_margin_rank_limit=config.selection_score_margin_rank_limit,
        cost_model=execution_context.cost_model,
        slippage_model=execution_context.slippage_model,
        exit_policy=execution_context.exit_policy,
        date_to_idx=pricing_context.date_to_idx,
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
        short_k_final = min(int(short_k_final), len(day) - int(config.rank_offset) - k)
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
