from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from .evaluation import (
    _build_period_positions as _build_period_positions_impl,
    _record_exposure_outputs,
    _record_period_backtest_outputs,
    _record_period_execution_sim,
    _record_period_ideal_daily_nav,
    _run_period_backtest,
)
from .portfolio_positions import build_positions_by_rebalance as build_positions_by_rebalance
from .position_postprocess import rebuild_backtest_from_positions
from .rebalance import _sample_rebalance_frame


def _build_period_positions(
    *,
    eval_df_full: pd.DataFrame,
    bt_rebalance: list[pd.Timestamp],
    context: Mapping[str, Any],
    allow_live_fallback: bool,
    build_positions_by_rebalance_fn=build_positions_by_rebalance,
) -> tuple[pd.DataFrame | None, dict[str, Any], dict[str, pd.DataFrame]]:
    return _build_period_positions_impl(
        eval_df_full=eval_df_full,
        bt_rebalance=bt_rebalance,
        context=context,
        allow_live_fallback=allow_live_fallback,
        build_positions_by_rebalance_fn=build_positions_by_rebalance_fn,
    )


def _record_period_backtest_nav_outputs(
    result: dict[str, Any],
    *,
    eval_df_full: pd.DataFrame,
    context: Mapping[str, Any],
    label_prefix: str,
    allow_live_fallback: bool,
    build_positions_by_rebalance_fn=build_positions_by_rebalance,
) -> Any:
    _, bt_rebalance = _sample_rebalance_frame(
        eval_df_full,
        frequency=context["backtest_rebalance_frequency"],
        valid_dates=context["valid_dates_set"],
    )
    result["backtest_rebalance_dates"] = bt_rebalance

    positions_by_rebalance, postprocess_meta, postprocess_artifacts = _build_period_positions(
        eval_df_full=eval_df_full,
        bt_rebalance=bt_rebalance,
        context=context,
        allow_live_fallback=allow_live_fallback,
        build_positions_by_rebalance_fn=build_positions_by_rebalance_fn,
    )
    result["positions_by_rebalance"] = positions_by_rebalance
    result["position_postprocess"] = postprocess_meta
    result["position_postprocess_artifacts"] = postprocess_artifacts

    bt_attempted, bt_result = _run_period_backtest(
        eval_df_full=eval_df_full,
        bt_rebalance=bt_rebalance,
        context=context,
        label_prefix=label_prefix,
    )
    bt_result = rebuild_backtest_from_positions(
        positions_by_rebalance,
        bt_result,
        context=context,
    )
    if bt_attempted:
        _record_period_backtest_outputs(
            result,
            bt_result=bt_result,
            context=context,
            label_prefix=label_prefix,
        )

    bt_period_info = bt_result[4] if bt_result is not None else None
    _record_period_ideal_daily_nav(
        result,
        positions_by_rebalance=positions_by_rebalance,
        period_info=bt_period_info,
        context=context,
        label_prefix=label_prefix,
    )
    _record_period_execution_sim(
        result,
        positions_by_rebalance=positions_by_rebalance,
        period_info=bt_period_info,
        context=context,
        label_prefix=label_prefix,
    )
    return positions_by_rebalance


def _record_period_exposure_outputs(
    result: dict[str, Any],
    *,
    eval_df_full: pd.DataFrame,
    positions_by_rebalance: Any,
    context: Mapping[str, Any],
) -> None:
    _record_exposure_outputs(
        result,
        eval_df_full=eval_df_full,
        exposure_source_df=context.get("exposure_source_df"),
        positions_by_rebalance=positions_by_rebalance,
        backtest_enabled=context["backtest_enabled"],
        backtest_pricing_df=context["backtest_pricing_df"],
        price_col=context["price_col"],
        benchmark_df=context["benchmark_df"],
        benchmark_return_series=context["benchmark_return_series"],
        fundamentals_mcap_col=context.get("fundamentals_mcap_col"),
        industry_columns=context.get("industry_columns", []),
        industry_source_df=context.get("industry_source_df"),
    )
