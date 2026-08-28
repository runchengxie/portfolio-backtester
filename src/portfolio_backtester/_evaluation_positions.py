"""Evaluation helpers for position filtering and execution-simulation recording."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import pandas as pd

from .execution import DetailedTradeFeeModel
from .execution_sim import (
    simulate_capacity_execution,
    simulate_execution_adjusted_nav,
    simulate_ideal_daily_nav,
)

logger = logging.getLogger("portfolio_backtester")


def _rebalance_key(value: Any) -> str | None:
    text = str(value).strip()
    compact = text.replace("-", "")
    if compact.endswith(".0"):
        compact = compact[:-2]
    if len(compact) == 8 and compact.isdigit():
        return compact
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return None
    resolved = cast(pd.Timestamp, pd.Timestamp(timestamp))
    return resolved.strftime("%Y%m%d")


def _filter_positions_to_backtest_periods(
    positions_by_rebalance: pd.DataFrame | None,
    period_info: list[Mapping[str, Any]] | None,
) -> pd.DataFrame | None:
    if positions_by_rebalance is None or positions_by_rebalance.empty or not period_info:
        return positions_by_rebalance
    if "rebalance_date" not in positions_by_rebalance.columns:
        return positions_by_rebalance

    executable_dates = {
        key
        for item in period_info
        for key in [_rebalance_key(item.get("rebalance_date"))]
        if key is not None
    }
    if not executable_dates:
        return positions_by_rebalance

    keys = positions_by_rebalance["rebalance_date"].map(_rebalance_key)
    return positions_by_rebalance.loc[keys.isin(list(executable_dates))].copy()


def _execution_trade_fee_model(context: Mapping[str, Any]) -> DetailedTradeFeeModel | None:
    model = context["execution_model"].cost_model
    return model if isinstance(model, DetailedTradeFeeModel) else None


def _record_period_execution_sim(
    result: dict[str, Any],
    *,
    positions_by_rebalance: pd.DataFrame | None,
    period_info: list[Mapping[str, Any]] | None = None,
    context: Mapping[str, Any],
    label_prefix: str,
) -> None:
    execution_sim_config = context["execution_sim_config"]
    if (
        not context["backtest_enabled"]
        or not getattr(execution_sim_config, "enabled", False)
        or positions_by_rebalance is None
        or positions_by_rebalance.empty
    ):
        return

    sim_positions = _filter_positions_to_backtest_periods(positions_by_rebalance, period_info)
    if sim_positions is None or sim_positions.empty:
        return
    if period_info:
        original_periods = positions_by_rebalance["rebalance_date"].map(_rebalance_key).nunique()
        aligned_periods = sim_positions["rebalance_date"].map(_rebalance_key).nunique()
        if aligned_periods < original_periods:
            logger.info(
                "%sExecution sim aligned to backtest periods: %d -> %d rebalances.",
                label_prefix,
                int(original_periods),
                int(aligned_periods),
            )

    backtest_pricing_df = context["backtest_pricing_df"]
    execution_model = context["execution_model"]
    tradable_col = context["backtest_tradable_col"]
    limit_up_col = _optional_market_rule_col(
        backtest_pricing_df,
        execution_sim_config.limit_up_col,
        "limit_up",
    )
    limit_down_col = _optional_market_rule_col(
        backtest_pricing_df,
        execution_sim_config.limit_down_col,
        "limit_down",
    )
    listing_status_col = _optional_market_rule_col(
        backtest_pricing_df,
        execution_sim_config.listing_status_col,
        "listing_status",
    )
    sim_result = simulate_capacity_execution(
        sim_positions,
        backtest_pricing_df,
        execution_sim_config,
        price_col=execution_model.entry_policy.price_col,
        tradable_col=tradable_col if tradable_col in backtest_pricing_df.columns else None,
        buy_tradable_col=(
            "is_buy_tradable" if "is_buy_tradable" in backtest_pricing_df.columns else None
        ),
        sell_tradable_col=(
            "is_sell_tradable" if "is_sell_tradable" in backtest_pricing_df.columns else None
        ),
        limit_up_col=limit_up_col,
        limit_down_col=limit_down_col,
        listing_status_col=listing_status_col,
    )
    result["execution_sim_summary"] = sim_result.summary
    result["execution_sim_orders"] = sim_result.orders
    result["execution_sim_fills"] = sim_result.fills
    executed_result = simulate_execution_adjusted_nav(
        sim_positions,
        backtest_pricing_df,
        execution_sim_config,
        price_col=execution_model.entry_policy.price_col,
        tradable_col=tradable_col if tradable_col in backtest_pricing_df.columns else None,
        buy_tradable_col=(
            "is_buy_tradable" if "is_buy_tradable" in backtest_pricing_df.columns else None
        ),
        sell_tradable_col=(
            "is_sell_tradable" if "is_sell_tradable" in backtest_pricing_df.columns else None
        ),
        limit_up_col=limit_up_col,
        limit_down_col=limit_down_col,
        listing_status_col=listing_status_col,
        transaction_cost_bps=context["backtest_cost_bps_effective"],
        trading_days_per_year=context["backtest_trading_days_per_year"],
        trade_fee_model=_execution_trade_fee_model(context),
    )
    result["execution_sim_executed_summary"] = executed_result.summary
    result["execution_sim_executed_daily"] = executed_result.daily
    if sim_result.summary.get("status") == "ok":
        logger.info(
            "%sExecution sim: fill ratio %.2f%%, unfilled %.2f",
            label_prefix,
            float(sim_result.summary.get("fill_ratio", np.nan)) * 100,
            float(sim_result.summary.get("unfilled_notional", 0.0)),
        )
    if executed_result.summary.get("status") == "ok":
        executed_stats = executed_result.summary.get("stats", {})
        logger.info(
            "%sExecution-adjusted NAV: total return %.2f%%, Sharpe %.2f",
            label_prefix,
            float(executed_stats.get("total_return", np.nan)) * 100,
            float(executed_stats.get("sharpe", np.nan)),
        )


def _record_period_ideal_daily_nav(
    result: dict[str, Any],
    *,
    positions_by_rebalance: pd.DataFrame | None,
    period_info: list[Mapping[str, Any]] | None,
    context: Mapping[str, Any],
    label_prefix: str,
) -> None:
    if (
        not context["backtest_enabled"]
        or positions_by_rebalance is None
        or positions_by_rebalance.empty
        or not period_info
    ):
        return

    nav_positions = _filter_positions_to_backtest_periods(positions_by_rebalance, period_info)
    if nav_positions is None or nav_positions.empty:
        return

    backtest_pricing_df = context["backtest_pricing_df"]
    execution_model = context["execution_model"]
    portfolio_value = float(
        getattr(context["execution_sim_config"], "portfolio_value", 1_000_000.0)
    )
    limit_up_col = _optional_market_rule_col(
        backtest_pricing_df,
        execution_sim_config.limit_up_col,
        "limit_up",
    )
    limit_down_col = _optional_market_rule_col(
        backtest_pricing_df,
        execution_sim_config.limit_down_col,
        "limit_down",
    )
    listing_status_col = _optional_market_rule_col(
        backtest_pricing_df,
        execution_sim_config.listing_status_col,
        "listing_status",
    )
    ideal_result = simulate_ideal_daily_nav(
        nav_positions,
        backtest_pricing_df,
        price_col=execution_model.entry_policy.price_col,
        limit_up_col=limit_up_col,
        limit_down_col=limit_down_col,
        listing_status_col=listing_status_col,
        transaction_cost_bps=context["backtest_cost_bps_effective"],
        trading_days_per_year=context["backtest_trading_days_per_year"],
        portfolio_value=portfolio_value,
        trade_fee_model=_execution_trade_fee_model(context),
    )
    result["ideal_daily_nav_summary"] = ideal_result.summary
    result["ideal_daily_nav_daily"] = ideal_result.daily
    result["ideal_daily_nav_orders"] = ideal_result.orders
    result["ideal_daily_nav_fills"] = ideal_result.fills
    if ideal_result.summary.get("status") == "ok":
        ideal_stats = ideal_result.summary.get("stats", {})
        logger.info(
            "%sIdeal daily NAV: total return %.2f%%, Sharpe %.2f",
            label_prefix,
            float(ideal_stats.get("total_return", np.nan)) * 100,
            float(ideal_stats.get("sharpe", np.nan)),
        )


def _optional_market_rule_col(
    pricing: pd.DataFrame,
    configured_name: str | None,
    default_name: str,
) -> str | None:
    """Resolve a configured market-rule column, with a legacy fallback.

    Returns the column name when present, else ``None``. The engine only
    consults a rule column when the corresponding rule is switched on, so a
    missing column is safe (the rule simply stays inactive).
    """
    if configured_name and configured_name in pricing.columns:
        return configured_name
    if default_name in pricing.columns:
        return default_name
    return None
