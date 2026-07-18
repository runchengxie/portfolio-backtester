"""Public score-driven backtest entry points."""

from __future__ import annotations

from typing import Literal

import pandas as pd

from .backtest_spec import BacktestSpec
from .contracts import GroupCap, StrategySpec
from .execution import (
    BpsCostModel,
    EntryPolicy,
    ExecutionModel,
    ExitPolicy,
    NoSlippageModel,
    SelectionConstraints,
)
from .selection_controls import MaxNewNamesShortfallPolicy


def run_backtest(
    data: pd.DataFrame,
    spec: BacktestSpec,
    *,
    pricing_data: pd.DataFrame | None = None,
):
    """Run a score-driven backtest from a composable specification.

    ``pricing_data`` is a runtime data input rather than part of ``BacktestSpec``
    so specifications remain safely serializable.
    """

    from .engine import _run_backtest_config
    from .topk_context import _build_backtest_spec_config

    config = _build_backtest_spec_config(spec, pricing_data=pricing_data)
    return _run_backtest_config(data, config=config)


def backtest_topk(
    data: pd.DataFrame,
    pred_col: str,
    price_col: str,
    rebalance_dates: list[pd.Timestamp],
    top_k: int,
    shift_days: int,
    cost_bps: float,
    trading_days_per_year: int,
    exit_mode: Literal["rebalance", "label_horizon"] = "rebalance",
    exit_horizon_days: int | None = None,
    long_only: bool = True,
    short_k: int | None = None,
    weighting: Literal["equal", "signal", "sqrt_liquidity"] = "equal",
    buffer_exit: int = 0,
    buffer_entry: int = 0,
    tradable_col: str | None = None,
    group_col: str | None = None,
    max_names_per_group: int | None = None,
    exit_price_policy: Literal["strict", "ffill", "delay"] = "strict",
    exit_fallback_policy: Literal["ffill", "none"] = "ffill",
    execution: ExecutionModel | None = None,
    pricing_data: pd.DataFrame | None = None,
    liquidity_floor_col: str | None = None,
    liquidity_floor_quantile: float | None = None,
    weighting_liquidity_col: str = "medadv20_amount",
    max_turnover_per_rebalance: float | None = None,
    rank_offset: int = 0,
    selection_tiebreak_col: str | None = None,
    selection_score_bucket_size: float | None = None,
    selection_score_margin: float | None = None,
    selection_score_margin_col: str | None = None,
    selection_score_margin_rank_limit: int | None = None,
    selection_min_score: float | None = None,
    max_new_names_per_rebalance: int | None = None,
    max_new_names_shortfall_policy: MaxNewNamesShortfallPolicy = "legacy_concentrate",
    max_positive_names: int | None = None,
):
    """Compatibility facade for the historical Top-K parameter surface."""

    if execution is None:
        if exit_price_policy not in {"strict", "ffill", "delay"}:
            raise ValueError("exit_price_policy must be one of: strict, ffill, delay.")
        if exit_fallback_policy not in {"ffill", "none"}:
            raise ValueError("exit_fallback_policy must be one of: ffill, none.")
        execution = ExecutionModel(
            cost_model=BpsCostModel(cost_bps),
            slippage_model=NoSlippageModel(),
            exit_policy=ExitPolicy(exit_price_policy, exit_fallback_policy, price_col),
            entry_policy=EntryPolicy(price_col),
            selection_constraints=SelectionConstraints(),
        )
    group_cap = None
    if group_col and max_names_per_group is not None:
        group_cap = GroupCap(column=group_col, max_names=max_names_per_group)
    strategy = StrategySpec(
        name=f"topk_k{top_k}",
        type="topk_buffered_long_only" if long_only else "topk_buffered_long_short",
        score_col=pred_col,
        top_k=top_k,
        buffer_exit=buffer_exit,
        buffer_entry=buffer_entry,
        weighting=weighting,
        long_only=long_only,
        short_k=short_k,
        group_cap=group_cap,
        source="backtest_topk_compatibility",
    )
    spec = BacktestSpec(
        strategy=strategy,
        execution=execution,
        rebalance_dates=tuple(rebalance_dates),
        shift_days=shift_days,
        trading_days_per_year=trading_days_per_year,
        exit_mode=exit_mode,
        exit_horizon_days=exit_horizon_days,
        tradable_col=tradable_col,
        liquidity_floor_col=liquidity_floor_col,
        liquidity_floor_quantile=liquidity_floor_quantile,
        weighting_liquidity_col=weighting_liquidity_col,
        max_turnover_per_rebalance=max_turnover_per_rebalance,
        rank_offset=rank_offset,
        selection_tiebreak_col=selection_tiebreak_col,
        selection_score_bucket_size=selection_score_bucket_size,
        selection_score_margin=selection_score_margin,
        selection_score_margin_col=selection_score_margin_col,
        selection_score_margin_rank_limit=selection_score_margin_rank_limit,
        selection_min_score=selection_min_score,
        max_new_names_per_rebalance=max_new_names_per_rebalance,
        max_new_names_shortfall_policy=max_new_names_shortfall_policy,
        max_positive_names=max_positive_names,
    )
    return run_backtest(data, spec, pricing_data=pricing_data)


__all__ = ["backtest_topk", "run_backtest"]
