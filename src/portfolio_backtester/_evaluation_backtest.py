"""Evaluation helpers for backtest recording and walk-forward orchestration."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from .benchmarking import build_benchmark_series, warn_if_delay_exit_lag as _warn_if_delay_exit_lag
from .engine import backtest_topk
from .exposure import compute_backtest_exposure_analysis
from .freshness_overlay import apply_freshness_overlay
from .metrics import summarize_active_returns, summarize_period_returns
from .portfolio_positions import build_positions_by_rebalance
from .position_postprocess import apply_position_postprocess
from .rebalance import get_rebalance_dates
from .signal_postprocess import apply_score_postprocess_inplace

logger = logging.getLogger("portfolio_backtester")


def _record_backtest_outputs(
    result: dict[str, Any],
    bt_result: tuple | None,
    *,
    label_prefix: str,
    backtest_long_only: bool,
    backtest_exit_mode: str,
    backtest_exit_price_policy: str,
    benchmark_df: pd.DataFrame | None,
    benchmark_return_series: pd.Series,
    execution_model: Any,
    backtest_trading_days_per_year: int,
) -> None:
    if bt_result is None:
        logger.info("%sBacktest not available - insufficient data.", label_prefix)
        return

    stats, net_series, gross_series, bt_turnover_series, period_info = bt_result
    result["bt_stats"] = stats
    result["bt_net_series"] = net_series
    result["bt_gross_series"] = gross_series
    result["bt_turnover_series"] = bt_turnover_series
    result["bt_periods"] = period_info
    mode_text = "long-only" if backtest_long_only else "long-short"
    logger.info(
        "%sBacktest (%s, top-K, exit_mode=%s):",
        label_prefix,
        mode_text,
        backtest_exit_mode,
    )
    logger.info("%s  periods: %s", label_prefix, stats["periods"])
    logger.info("%s  total return: %.2f%%", label_prefix, stats["total_return"] * 100)
    logger.info("%s  ann return: %.2f%%", label_prefix, stats["ann_return"] * 100)
    logger.info("%s  ann vol: %.2f%%", label_prefix, stats["ann_vol"] * 100)
    logger.info("%s  sharpe: %.2f", label_prefix, stats["sharpe"])
    logger.info("%s  max drawdown: %.2f%%", label_prefix, stats["max_drawdown"] * 100)
    if not np.isnan(stats["avg_turnover"]):
        logger.info("%s  avg turnover: %.2f%%", label_prefix, stats["avg_turnover"] * 100)
        logger.info(
            "%s  avg cost drag: %.2f%%",
            label_prefix,
            stats["avg_cost_drag"] * 100,
        )
    _warn_if_delay_exit_lag(
        label_prefix=label_prefix,
        exit_price_policy=backtest_exit_price_policy,
        stats=stats,
    )

    bench_series, bench_periods = build_benchmark_series(
        benchmark_df,
        execution_model.entry_policy.price_col,
        execution_model.exit_policy.price_col,
        period_info,
        benchmark_return_series=benchmark_return_series,
    )
    if bench_series.empty:
        return
    result["bt_benchmark_series"] = bench_series
    bt_benchmark_stats = summarize_period_returns(
        bench_series, bench_periods, backtest_trading_days_per_year
    )
    result["bt_benchmark_stats"] = bt_benchmark_stats
    logger.info(
        "%s  benchmark total return: %.2f%%",
        label_prefix,
        bt_benchmark_stats["total_return"] * 100,
    )
    periods_per_year = stats.get("periods_per_year", np.nan)
    bt_active_stats, bt_active_series = summarize_active_returns(
        net_series, bench_series, periods_per_year
    )
    result["bt_active_stats"] = bt_active_stats
    result["bt_active_series"] = bt_active_series
    if bt_active_stats and bt_active_stats.get("n", 0) > 0:
        logger.info(
            "%s  active total return: %.2f%%",
            label_prefix,
            bt_active_stats["active_total_return"] * 100,
        )
        if np.isfinite(bt_active_stats.get("information_ratio", np.nan)):
            logger.info(
                "%s  information ratio: %.2f",
                label_prefix,
                bt_active_stats["information_ratio"],
            )
        if np.isfinite(bt_active_stats.get("beta", np.nan)):
            logger.info("%s  beta: %.2f", label_prefix, bt_active_stats["beta"])
        if np.isfinite(bt_active_stats.get("alpha", np.nan)):
            logger.info(
                "%s  alpha (ann): %.2f%%",
                label_prefix,
                bt_active_stats["alpha"] * 100,
            )


def _record_exposure_outputs(
    result: dict[str, Any],
    *,
    eval_df_full: pd.DataFrame,
    exposure_source_df: pd.DataFrame | None,
    positions_by_rebalance: pd.DataFrame | None,
    backtest_enabled: bool,
    backtest_pricing_df: pd.DataFrame,
    price_col: str,
    benchmark_df: pd.DataFrame | None,
    benchmark_return_series: pd.Series,
    fundamentals_mcap_col: str | None,
    industry_columns: list[str],
    industry_source_df: pd.DataFrame | None,
) -> None:
    if not backtest_enabled:
        return
    if positions_by_rebalance is None or positions_by_rebalance.empty:
        return
    exposure = compute_backtest_exposure_analysis(
        exposure_source_df if exposure_source_df is not None else eval_df_full,
        positions_by_rebalance,
        pricing_data=backtest_pricing_df,
        price_col=price_col,
        benchmark_df=benchmark_df,
        benchmark_return_series=benchmark_return_series,
        market_cap_col=fundamentals_mcap_col,
        industry_columns=industry_columns,
        industry_source_data=industry_source_df,
    )
    result["bt_style_exposure"] = exposure["style"]
    result["bt_style_exposure_summary"] = exposure["style_summary"]
    result["bt_industry_exposure"] = exposure["industry"]
    result["bt_industry_exposure_summary"] = exposure["industry_summary"]
    result["bt_active_exposure_summary"] = exposure["active_summary"]


def _score_walk_forward_backtest_frame(
    window_meta: Mapping[str, Any],
    *,
    model_w: Any,
    direction: float,
    context: Mapping[str, Any],
) -> tuple[pd.DataFrame, str] | None:
    bt_direction = (
        direction
        if context["backtest_signal_direction_raw"] is None
        else context["backtest_signal_direction_raw"]
    )
    bt_pred_col = "pred"
    test_start = pd.to_datetime(window_meta["test_start"])
    test_end = pd.to_datetime(window_meta["test_end"])
    df_full = context["df_full"]
    test_full_w = df_full[
        (df_full["trade_date"] >= test_start) & (df_full["trade_date"] <= test_end)
    ].copy()
    if test_full_w.empty:
        return None

    features = context["features"]
    test_full_w["pred"] = model_w.predict(test_full_w[features])
    apply_score_postprocess_inplace(
        test_full_w,
        "pred",
        method=context["score_postprocess_method"],
        columns=context["score_postprocess_columns"],
        strength=context["score_postprocess_strength"],
        min_obs=context["score_postprocess_min_obs"],
    )
    if bt_direction != 1.0:
        test_full_w["signal_bt"] = test_full_w["pred"] * bt_direction
        bt_pred_col = "signal_bt"
    test_full_w, _ = apply_freshness_overlay(
        test_full_w,
        score_col=bt_pred_col,
        cfg=context.get("freshness_overlay"),
    )
    return test_full_w, bt_pred_col


def _run_walk_forward_backtest_topk(
    test_full_w: pd.DataFrame,
    *,
    bt_pred_col: str,
    context: Mapping[str, Any],
    valid_dates_set: set[pd.Timestamp],
    backtest_topk_fn,
):
    bt_rebalance = get_rebalance_dates(
        sorted(test_full_w["trade_date"].unique()),
        context["backtest_rebalance_frequency"],
    )
    if valid_dates_set:
        bt_rebalance = [date for date in bt_rebalance if date in valid_dates_set]

    backtest_pricing_df = context["backtest_pricing_df"]
    backtest_tradable_col = context["backtest_tradable_col"]
    try:
        return backtest_topk_fn(
            test_full_w,
            pred_col=bt_pred_col,
            price_col=context["price_col"],
            rebalance_dates=bt_rebalance,
            top_k=context["backtest_top_k"],
            shift_days=context["label_shift_days"],
            cost_bps=context["backtest_cost_bps_effective"],
            trading_days_per_year=context["backtest_trading_days_per_year"],
            weighting=context["backtest_weighting"],
            exit_mode=context["backtest_exit_mode"],
            exit_horizon_days=context["backtest_exit_horizon_days"],
            long_only=context["backtest_long_only"],
            short_k=context["backtest_short_k"],
            buffer_exit=context["backtest_buffer_exit"],
            buffer_entry=context["backtest_buffer_entry"],
            group_col=(
                context["backtest_group_col"]
                if context["backtest_group_col"] in test_full_w.columns
                else None
            ),
            max_names_per_group=context["backtest_max_names_per_group"],
            tradable_col=(
                backtest_tradable_col
                if backtest_tradable_col in backtest_pricing_df.columns
                else None
            ),
            exit_price_policy=context["backtest_exit_price_policy"],
            exit_fallback_policy=context["backtest_exit_fallback_policy"],
            execution=context["execution_model"],
            pricing_data=backtest_pricing_df,
            liquidity_floor_col=context.get("backtest_liquidity_floor_col"),
            liquidity_floor_quantile=context.get("backtest_liquidity_floor_quantile"),
            weighting_liquidity_col=context.get(
                "backtest_weighting_liquidity_col",
                "medadv20_amount",
            ),
            max_turnover_per_rebalance=context.get("backtest_max_turnover_per_rebalance"),
            selection_tiebreak_col=context.get("backtest_selection_tiebreak_col"),
            selection_score_bucket_size=context.get("backtest_selection_score_bucket_size"),
            selection_score_margin=context.get("backtest_selection_score_margin"),
            selection_score_margin_col=context.get("backtest_selection_score_margin_col"),
            selection_score_margin_rank_limit=context.get(
                "backtest_selection_score_margin_rank_limit"
            ),
            selection_min_score=context.get("backtest_selection_min_score"),
            max_new_names_per_rebalance=context.get("backtest_max_new_names_per_rebalance"),
            max_new_names_shortfall_policy=context.get(
                "backtest_max_new_names_shortfall_policy",
                "legacy_concentrate",
            ),
            max_positive_names=context.get("backtest_max_positive_names"),
        )
    except ValueError:
        return None


def _summarize_walk_forward_benchmark(
    bt_result_w: Any,
    *,
    context: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    bt_stats_w, bt_net_w, _, _, bt_periods_w = bt_result_w
    execution_model = context["execution_model"]
    bench_series_w, bench_periods_w = build_benchmark_series(
        context["benchmark_df"],
        execution_model.entry_policy.price_col,
        execution_model.exit_policy.price_col,
        bt_periods_w,
        benchmark_return_series=context["benchmark_return_series"],
    )
    if bench_series_w.empty:
        return bt_stats_w, None, None

    bt_benchmark_stats_w = summarize_period_returns(
        bench_series_w,
        bench_periods_w,
        context["backtest_trading_days_per_year"],
    )
    periods_per_year = bt_stats_w.get("periods_per_year", np.nan)
    bt_active_stats_w, _ = summarize_active_returns(bt_net_w, bench_series_w, periods_per_year)
    return bt_stats_w, bt_benchmark_stats_w, bt_active_stats_w


def _evaluate_walk_forward_backtest(
    window_meta: Mapping[str, Any],
    *,
    model_w: Any,
    direction: float,
    context: Mapping[str, Any],
    valid_dates_set: set[pd.Timestamp],
    backtest_topk_fn,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    if not context["wf_backtest_enabled"]:
        return None, None, None

    scored = _score_walk_forward_backtest_frame(
        window_meta,
        model_w=model_w,
        direction=direction,
        context=context,
    )
    if scored is None:
        return None, None, None
    test_full_w, bt_pred_col = scored
    bt_result_w = _run_walk_forward_backtest_topk(
        test_full_w,
        bt_pred_col=bt_pred_col,
        context=context,
        valid_dates_set=valid_dates_set,
        backtest_topk_fn=backtest_topk_fn,
    )
    if bt_result_w is None:
        return None, None, None
    return _summarize_walk_forward_benchmark(bt_result_w, context=context)


def _build_period_positions(
    *,
    eval_df_full: pd.DataFrame,
    bt_rebalance: list[pd.Timestamp],
    context: Mapping[str, Any],
    allow_live_fallback: bool,
    build_positions_by_rebalance_fn: Callable[..., pd.DataFrame] = build_positions_by_rebalance,
) -> tuple[pd.DataFrame | None, dict[str, Any], dict[str, pd.DataFrame]]:
    backtest_enabled = context["backtest_enabled"]
    live_enabled = context["live_enabled"]
    positions_by_rebalance = None
    if backtest_enabled or not live_enabled or not allow_live_fallback:
        tradable_col = context["backtest_tradable_col"]
        group_col = context["backtest_group_col"]
        positions_by_rebalance = build_positions_by_rebalance_fn(
            eval_df_full,
            pred_col="signal_backtest",
            price_col=context["price_col"],
            rebalance_dates=bt_rebalance,
            top_k=context["backtest_top_k"],
            shift_days=context["label_shift_days"],
            weighting=context["backtest_weighting"],
            buffer_exit=context["backtest_buffer_exit"],
            buffer_entry=context["backtest_buffer_entry"],
            long_only=context["backtest_long_only"],
            short_k=context["backtest_short_k"],
            tradable_col=tradable_col if tradable_col in eval_df_full.columns else None,
            group_col=group_col if group_col in eval_df_full.columns else None,
            max_names_per_group=context["backtest_max_names_per_group"],
            execution=context["execution_model"],
            pricing_data=context["backtest_pricing_df"],
            liquidity_floor_col=context.get("backtest_liquidity_floor_col"),
            liquidity_floor_quantile=context.get("backtest_liquidity_floor_quantile"),
            weighting_liquidity_col=context.get(
                "backtest_weighting_liquidity_col",
                "medadv20_amount",
            ),
            max_turnover_per_rebalance=context.get("backtest_max_turnover_per_rebalance"),
            selection_tiebreak_col=context.get("backtest_selection_tiebreak_col"),
            selection_score_bucket_size=context.get("backtest_selection_score_bucket_size"),
            selection_score_margin=context.get("backtest_selection_score_margin"),
            selection_score_margin_col=context.get("backtest_selection_score_margin_col"),
            selection_score_margin_rank_limit=context.get(
                "backtest_selection_score_margin_rank_limit"
            ),
            selection_min_score=context.get("backtest_selection_min_score"),
            max_new_names_per_rebalance=context.get("backtest_max_new_names_per_rebalance"),
            max_new_names_shortfall_policy=context.get(
                "backtest_max_new_names_shortfall_policy",
                "legacy_concentrate",
            ),
            max_positive_names=context.get("backtest_max_positive_names"),
        )
    if allow_live_fallback and live_enabled and not backtest_enabled:
        positions_by_rebalance = context["positions_by_rebalance_live"]
    positions_by_rebalance, postprocess_meta, postprocess_artifacts = apply_position_postprocess(
        positions_by_rebalance,
        eval_df_full=eval_df_full,
        context=context,
    )
    return positions_by_rebalance, postprocess_meta, postprocess_artifacts


def _run_period_backtest(
    *,
    eval_df_full: pd.DataFrame,
    bt_rebalance: list[pd.Timestamp],
    context: Mapping[str, Any],
    label_prefix: str,
) -> tuple[bool, Any | None]:
    if not context["backtest_enabled"]:
        return False, None

    backtest_pricing_df = context["backtest_pricing_df"]
    tradable_col = context["backtest_tradable_col"]
    group_col = context["backtest_group_col"]
    try:
        return True, context.get("backtest_topk_fn", backtest_topk)(
            eval_df_full,
            pred_col="signal_backtest",
            price_col=context["price_col"],
            rebalance_dates=bt_rebalance,
            top_k=context["backtest_top_k"],
            shift_days=context["label_shift_days"],
            cost_bps=context["backtest_cost_bps_effective"],
            trading_days_per_year=context["backtest_trading_days_per_year"],
            weighting=context["backtest_weighting"],
            exit_mode=context["backtest_exit_mode"],
            exit_horizon_days=context["backtest_exit_horizon_days"],
            long_only=context["backtest_long_only"],
            short_k=context["backtest_short_k"],
            buffer_exit=context["backtest_buffer_exit"],
            buffer_entry=context["backtest_buffer_entry"],
            group_col=group_col if group_col in eval_df_full.columns else None,
            max_names_per_group=context["backtest_max_names_per_group"],
            tradable_col=tradable_col if tradable_col in backtest_pricing_df.columns else None,
            exit_price_policy=context["backtest_exit_price_policy"],
            exit_fallback_policy=context["backtest_exit_fallback_policy"],
            execution=context["execution_model"],
            pricing_data=backtest_pricing_df,
            liquidity_floor_col=context.get("backtest_liquidity_floor_col"),
            liquidity_floor_quantile=context.get("backtest_liquidity_floor_quantile"),
            weighting_liquidity_col=context.get(
                "backtest_weighting_liquidity_col",
                "medadv20_amount",
            ),
            max_turnover_per_rebalance=context.get("backtest_max_turnover_per_rebalance"),
            selection_tiebreak_col=context.get("backtest_selection_tiebreak_col"),
            selection_score_bucket_size=context.get("backtest_selection_score_bucket_size"),
            selection_score_margin=context.get("backtest_selection_score_margin"),
            selection_score_margin_col=context.get("backtest_selection_score_margin_col"),
            selection_score_margin_rank_limit=context.get(
                "backtest_selection_score_margin_rank_limit"
            ),
            selection_min_score=context.get("backtest_selection_min_score"),
            max_new_names_per_rebalance=context.get("backtest_max_new_names_per_rebalance"),
            max_new_names_shortfall_policy=context.get(
                "backtest_max_new_names_shortfall_policy",
                "legacy_concentrate",
            ),
            max_positive_names=context.get("backtest_max_positive_names"),
        )
    except ValueError as exc:
        logger.warning("%sBacktest skipped: %s", label_prefix, exc)
        return True, None


def _record_period_backtest_outputs(
    result: dict[str, Any],
    *,
    bt_result: Any | None,
    context: Mapping[str, Any],
    label_prefix: str,
) -> None:
    _record_backtest_outputs(
        result,
        bt_result,
        label_prefix=label_prefix,
        backtest_long_only=context["backtest_long_only"],
        backtest_exit_mode=context["backtest_exit_mode"],
        backtest_exit_price_policy=context["backtest_exit_price_policy"],
        benchmark_df=context["benchmark_df"],
        benchmark_return_series=context["benchmark_return_series"],
        execution_model=context["execution_model"],
        backtest_trading_days_per_year=context["backtest_trading_days_per_year"],
    )
