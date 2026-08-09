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
from .selection_controls import (
    MaxNewNamesShortfallPolicy,
    SelectionPricePolicy,
    TargetWeightPolicy,
)


def run_backtest(
    data: pd.DataFrame,
    spec: BacktestSpec,
    *,
    pricing_data: pd.DataFrame | None = None,
    ledger: bool = False,
    ledger_config: object | None = None,
):
    """Run a score-driven backtest from a composable specification.

    ``pricing_data`` is a runtime data input rather than part of ``BacktestSpec``
    so specifications remain safely serializable. When ``ledger`` is True, the
    returned bundle gains a trailing ``UnifiedLedger`` element produced by the
    shared execution-sim engine; the default ``ledger=False`` preserves the
    historical five-element return contract.
    """

    from .engine import _run_backtest_config
    from .topk_context import _build_backtest_spec_config

    config = _build_backtest_spec_config(spec, pricing_data=pricing_data)
    return _run_backtest_config(data, config=config, ledger=ledger, ledger_config=ledger_config)


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
    entry_rank_cutoff: int | None = None,
    selection_price_policy: SelectionPricePolicy = "execution_aware",
    target_weight_policy: TargetWeightPolicy = "normalized",
    ledger: bool = False,
    ledger_config: object | None = None,
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
    spec = _assemble_topk_spec(
        execution=execution,
        pred_col=pred_col,
        top_k=top_k,
        long_only=long_only,
        buffer_exit=buffer_exit,
        buffer_entry=buffer_entry,
        weighting=weighting,
        short_k=short_k,
        group_col=group_col,
        max_names_per_group=max_names_per_group,
        rebalance_dates=rebalance_dates,
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
        entry_rank_cutoff=entry_rank_cutoff,
        selection_price_policy=selection_price_policy,
        target_weight_policy=target_weight_policy,
    )
    return run_backtest(
        data, spec, pricing_data=pricing_data, ledger=ledger, ledger_config=ledger_config
    )


def _assemble_topk_spec(
    *,
    execution: ExecutionModel,
    pred_col: str,
    top_k: int,
    long_only: bool,
    buffer_exit: int,
    buffer_entry: int,
    weighting: str,
    short_k: int | None,
    group_col: str | None,
    max_names_per_group: int | None,
    rebalance_dates: list[pd.Timestamp],
    shift_days: int,
    trading_days_per_year: int,
    exit_mode: str,
    exit_horizon_days: int | None,
    tradable_col: str | None,
    liquidity_floor_col: str | None,
    liquidity_floor_quantile: float | None,
    weighting_liquidity_col: str,
    max_turnover_per_rebalance: float | None,
    rank_offset: int,
    selection_tiebreak_col: str | None,
    selection_score_bucket_size: float | None,
    selection_score_margin: float | None,
    selection_score_margin_col: str | None,
    selection_score_margin_rank_limit: int | None,
    selection_min_score: float | None,
    max_new_names_per_rebalance: int | None,
    max_new_names_shortfall_policy: object,
    max_positive_names: int | None,
    entry_rank_cutoff: int | None,
    selection_price_policy: object,
    target_weight_policy: object,
) -> BacktestSpec:
    """Compose the BacktestSpec used by :func:`backtest_topk`."""
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
    return BacktestSpec(
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
        entry_rank_cutoff=entry_rank_cutoff,
        selection_price_policy=selection_price_policy,
        target_weight_policy=target_weight_policy,
    )


__all__ = ["backtest_topk", "run_backtest"]
