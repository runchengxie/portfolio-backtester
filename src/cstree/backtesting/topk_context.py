from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, cast

import numpy as np
import pandas as pd

from .backtest_spec import BacktestSpec
from .execution import ExecutionModel
from .metrics import summarize_period_returns
from .portfolio_weights import normalize_weighting_mode
from .pricing import (
    normalize_backtest_frame,
    prepare_backtest_pricing_context,
    resolve_backtest_execution_context,
)
from .types import (
    BacktestExecutionContext,
    BacktestPeriodResult,
    BacktestPositionState,
    BacktestPricingContext,
)


@dataclass(frozen=True)
class _BacktestTopKConfig:
    pred_col: str
    price_col: str
    rebalance_dates: list[pd.Timestamp]
    top_k: int
    rank_offset: int
    shift_days: int
    cost_bps: float
    trading_days_per_year: int
    exit_mode: Literal["rebalance", "label_horizon"]
    exit_horizon_days: int | None
    long_only: bool
    short_k: int | None
    weighting: Literal["equal", "signal", "sqrt_liquidity"]
    buffer_exit: int
    buffer_entry: int
    tradable_col: str | None
    group_col: str | None
    max_names_per_group: int | None
    exit_price_policy: Literal["strict", "ffill", "delay"]
    exit_fallback_policy: Literal["ffill", "none"]
    execution: ExecutionModel | None
    pricing_data: pd.DataFrame | None
    liquidity_floor_col: str | None
    liquidity_floor_quantile: float | None
    weighting_liquidity_col: str
    max_turnover_per_rebalance: float | None
    selection_tiebreak_col: str | None
    selection_score_bucket_size: float | None
    selection_score_margin: float | None
    selection_score_margin_rank_limit: int | None


@dataclass(frozen=True)
class _BacktestRunContext:
    execution_context: BacktestExecutionContext
    pricing_context: BacktestPricingContext
    weighting_mode: str


@dataclass
class _BacktestResultAccumulator:
    net_returns: list[float] = field(default_factory=list)
    gross_returns: list[float] = field(default_factory=list)
    turnovers: list[float] = field(default_factory=list)
    costs: list[float] = field(default_factory=list)
    fee_costs: list[float] = field(default_factory=list)
    slippage_costs: list[float] = field(default_factory=list)
    period_info: list[dict] = field(default_factory=list)
    long_state: BacktestPositionState = field(default_factory=BacktestPositionState)
    short_state: BacktestPositionState = field(default_factory=BacktestPositionState)
    prev_exit_idx: int | None = None


@dataclass(frozen=True)
class _BacktestPeriodEvaluation:
    period_result: BacktestPeriodResult
    reb_date: pd.Timestamp
    entry_idx: int
    planned_exit_idx: int
    entry_date: pd.Timestamp
    planned_exit_date: pd.Timestamp
    long_state: BacktestPositionState
    short_state: BacktestPositionState


def _build_backtest_spec_config(
    spec: BacktestSpec,
    *,
    pricing_data: pd.DataFrame | None,
) -> _BacktestTopKConfig:
    strategy = spec.strategy
    group_cap = strategy.group_cap
    execution = spec.execution
    return _BacktestTopKConfig(
        pred_col=strategy.score_col,
        price_col=execution.entry_policy.price_col,
        rebalance_dates=list(spec.rebalance_dates),
        top_k=strategy.top_k,
        rank_offset=spec.rank_offset,
        shift_days=spec.shift_days,
        cost_bps=0.0,
        trading_days_per_year=spec.trading_days_per_year,
        exit_mode=spec.exit_mode,
        exit_horizon_days=spec.exit_horizon_days,
        long_only=strategy.long_only,
        short_k=strategy.short_k,
        weighting=cast(Literal["equal", "signal", "sqrt_liquidity"], strategy.weighting),
        buffer_exit=strategy.buffer_exit,
        buffer_entry=strategy.buffer_entry,
        tradable_col=spec.tradable_col,
        group_col=group_cap.column if group_cap is not None else None,
        max_names_per_group=group_cap.max_names if group_cap is not None else None,
        exit_price_policy=execution.exit_policy.price_policy,
        exit_fallback_policy=execution.exit_policy.fallback_policy,
        execution=execution,
        pricing_data=pricing_data,
        liquidity_floor_col=spec.liquidity_floor_col,
        liquidity_floor_quantile=spec.liquidity_floor_quantile,
        weighting_liquidity_col=spec.weighting_liquidity_col,
        max_turnover_per_rebalance=spec.max_turnover_per_rebalance,
        selection_tiebreak_col=spec.selection_tiebreak_col,
        selection_score_bucket_size=spec.selection_score_bucket_size,
        selection_score_margin=spec.selection_score_margin,
        selection_score_margin_rank_limit=spec.selection_score_margin_rank_limit,
    )


def _merge_backtest_supplemental_columns(
    data: pd.DataFrame | None,
    *,
    pricing_source: pd.DataFrame | None,
    config: _BacktestTopKConfig,
) -> pd.DataFrame | None:
    if data is None or pricing_source is None:
        return data
    supplemental_cols = [
        col
        for col in {config.liquidity_floor_col, config.weighting_liquidity_col}
        if col and col not in data.columns and col in pricing_source.columns
    ]
    if not supplemental_cols:
        return data
    return data.merge(
        pricing_source[["trade_date", "symbol", *supplemental_cols]],
        on=["trade_date", "symbol"],
        how="left",
    )


def _prepare_backtest_run_context(
    data: pd.DataFrame,
    *,
    config: _BacktestTopKConfig,
) -> _BacktestRunContext | None:
    normalized_data = normalize_backtest_frame(data, context="Backtest data")
    pricing_data = normalize_backtest_frame(
        config.pricing_data,
        context="Backtest pricing data",
    )
    pricing_source_for_supplements = pricing_data if pricing_data is not None else normalized_data
    normalized_data = _merge_backtest_supplemental_columns(
        normalized_data,
        pricing_source=pricing_source_for_supplements,
        config=config,
    )
    execution_context = resolve_backtest_execution_context(
        execution=config.execution,
        exit_price_policy=config.exit_price_policy,
        exit_fallback_policy=config.exit_fallback_policy,
        price_col=config.price_col,
        cost_bps=config.cost_bps,
    )
    weighting_mode = normalize_weighting_mode(config.weighting)
    pricing_context = prepare_backtest_pricing_context(
        data=normalized_data,
        pricing_data=pricing_data,
        entry_policy=execution_context.entry_policy,
        exit_policy=execution_context.exit_policy,
        selection_constraints=execution_context.selection_constraints,
        slippage_model=execution_context.slippage_model,
        tradable_col=config.tradable_col,
    )
    if pricing_context is None:
        return None
    return _BacktestRunContext(
        execution_context=execution_context,
        pricing_context=pricing_context,
        weighting_mode=weighting_mode,
    )


def _build_backtest_return_bundle(
    *,
    accumulator: _BacktestResultAccumulator,
    config: _BacktestTopKConfig,
    weighting_mode: str,
):
    index = [info["exit_date"] for info in accumulator.period_info]
    net_series = pd.Series(accumulator.net_returns, index=index, name="net_return")
    gross_series = pd.Series(accumulator.gross_returns, index=index, name="gross_return")
    turnover_series = pd.Series(accumulator.turnovers, index=index, name="turnover")

    stats = summarize_period_returns(
        net_series,
        accumulator.period_info,
        config.trading_days_per_year,
    )
    avg_turnover = turnover_series.dropna().mean() if turnover_series.notna().any() else np.nan
    avg_cost = float(np.mean(accumulator.costs)) if accumulator.costs else np.nan
    avg_fee_cost = float(np.mean(accumulator.fee_costs)) if accumulator.fee_costs else np.nan
    avg_slippage_cost = (
        float(np.mean(accumulator.slippage_costs)) if accumulator.slippage_costs else np.nan
    )
    stats.update(
        {
            "avg_turnover": avg_turnover,
            "avg_cost_drag": avg_cost,
            "avg_fee_drag": avg_fee_cost,
            "avg_slippage_drag": avg_slippage_cost,
            "mode": "long_only" if config.long_only else "long_short",
            "weighting": weighting_mode,
            "long_k": int(config.top_k),
            "rank_offset": int(config.rank_offset),
            "short_k": int(config.short_k)
            if (not config.long_only and config.short_k is not None)
            else None,
        }
    )
    return stats, net_series, gross_series, turnover_series, accumulator.period_info
