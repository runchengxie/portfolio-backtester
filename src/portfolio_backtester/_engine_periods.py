"""Backtest engine: period evaluation, accumulation, and config entry point."""

from __future__ import annotations

from typing import cast

import pandas as pd

from ._engine_leg import (
    _apply_liquidity_floor,
    _BacktestLegContext,
    _evaluate_long_only_period,
    _evaluate_long_short_period,
)
from .execution_sim import ExecutionSimConfig
from .period_turnover import period_turnover_fields
from .periods import resolve_backtest_period_plan
from .topk_context import (
    _BacktestPeriodEvaluation,
    _BacktestResultAccumulator,
    _BacktestRunContext,
    _BacktestTopKConfig,
    _build_backtest_return_bundle,
    _prepare_backtest_run_context,
)
from .types import BacktestPeriodResult, BacktestPositionState


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
        entry_rank_cutoff=config.entry_rank_cutoff,
        selection_price_policy=config.selection_price_policy,
        target_weight_policy=config.target_weight_policy,
        target_slot_count=config.top_k,
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
        accumulator.targets_by_rebalance.append(
            {
                "rebalance_date": evaluation.reb_date,
                "entry_date": evaluation.entry_date,
                "long_state": evaluation.long_state,
                "short_state": evaluation.short_state,
            }
        )
        accumulator.long_state = evaluation.long_state
        accumulator.short_state = evaluation.short_state
        accumulator.prev_exit_idx = evaluation.period_result.exit_idx
    return accumulator


def _run_backtest_config(
    data: pd.DataFrame,
    *,
    config: _BacktestTopKConfig,
    ledger: bool = False,
    ledger_config: object | None = None,
):
    run_context = _prepare_backtest_run_context(data, config=config)
    if run_context is None:
        return None
    accumulator = _run_backtest_periods(config=config, run_context=run_context)
    if not accumulator.net_returns:
        return None
    bundle = _build_backtest_return_bundle(
        accumulator=accumulator,
        config=config,
        weighting_mode=run_context.weighting_mode,
    )
    if not ledger:
        return bundle
    from .execution_sim import simulate_ideal_daily_nav

    sim_config = _resolve_ledger_config(ledger_config)
    positions = _build_ledger_positions(accumulator)
    pricing = config.pricing_data
    if positions is None or positions.empty or pricing is None:
        return (*bundle, None)
    ledger_result = simulate_ideal_daily_nav(
        positions,
        pricing,
        price_col=config.price_col,
        transaction_cost_bps=_effective_cost_bps(config),
        trading_days_per_year=config.trading_days_per_year,
        portfolio_value=float(sim_config.portfolio_value),
    )
    unified = ledger_result.to_unified_ledger(portfolio_value=float(sim_config.portfolio_value))
    return (*bundle, unified)


def _resolve_ledger_config(ledger_config: object | None) -> ExecutionSimConfig:
    from .execution_sim import ExecutionSimConfig, build_execution_sim_config

    if ledger_config is None:
        return ExecutionSimConfig(enabled=True)
    if isinstance(ledger_config, ExecutionSimConfig):
        return ledger_config
    return build_execution_sim_config(ledger_config)


def _effective_cost_bps(config: _BacktestTopKConfig) -> float:
    execution = config.execution
    if execution is not None and execution.cost_model is not None:
        cost_model = execution.cost_model
        bps = getattr(cost_model, "bps", None)
        if bps is not None:
            return float(bps)
    return 0.0


def _build_ledger_positions(accumulator: object) -> pd.DataFrame | None:
    from .types import BacktestPositionState

    targets = getattr(accumulator, "targets_by_rebalance", None)
    if not targets:
        return None
    rows = []
    for item in targets:
        rebalance_date = item["rebalance_date"]
        entry_date = item["entry_date"]
        long_state = item["long_state"]
        short_state = item["short_state"]
        long_weights = (
            long_state.target_weights if isinstance(long_state, BacktestPositionState) else None
        )
        if long_weights is not None and len(long_weights) > 0:
            for symbol, weight in long_weights.items():
                rows.append(
                    {
                        "rebalance_date": rebalance_date,
                        "entry_date": entry_date,
                        "symbol": str(symbol),
                        "weight": float(weight),
                        "side": "long",
                    }
                )
        short_weights = (
            short_state.target_weights if isinstance(short_state, BacktestPositionState) else None
        )
        if short_weights is not None and len(short_weights) > 0:
            for symbol, weight in short_weights.items():
                rows.append(
                    {
                        "rebalance_date": rebalance_date,
                        "entry_date": entry_date,
                        "symbol": str(symbol),
                        "weight": float(-abs(weight)),
                        "side": "short",
                    }
                )
    if not rows:
        return None
    return pd.DataFrame(rows)
