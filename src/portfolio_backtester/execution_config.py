"""Shared execution configuration normalization for portfolio backtests."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .execution import BpsCostModel, build_execution_model, required_pricing_columns
from .execution_sim import build_execution_sim_config, required_execution_sim_columns


def _merge_execution_config(
    *,
    execution_cfg: Mapping[str, Any],
    backtest_cfg: Mapping[str, Any],
) -> Mapping[str, Any]:
    backtest_execution_cfg = backtest_cfg.get("execution")
    if not isinstance(backtest_execution_cfg, Mapping):
        return execution_cfg

    merged_execution_cfg = dict(execution_cfg)
    for key, value in backtest_execution_cfg.items():
        if isinstance(value, Mapping) and isinstance(merged_execution_cfg.get(key), Mapping):
            nested = dict(merged_execution_cfg[key])
            nested.update(value)
            merged_execution_cfg[key] = nested
        else:
            merged_execution_cfg[key] = value
    return merged_execution_cfg


def resolve_execution_settings(
    *,
    execution_cfg: Mapping[str, Any],
    backtest_cfg: Mapping[str, Any],
    backtest_settings: Mapping[str, Any],
    price_col: str,
) -> dict[str, Any]:
    execution_cfg_resolved = _merge_execution_config(
        execution_cfg=execution_cfg,
        backtest_cfg=backtest_cfg,
    )
    execution_source = (
        "explicit_execution_config" if bool(execution_cfg_resolved) else "default_flat_cost"
    )
    exit_price_policy = backtest_settings["BACKTEST_EXIT_PRICE_POLICY"]
    if exit_price_policy not in {"strict", "ffill", "delay"}:
        raise SystemExit("backtest.exit_price_policy must be one of: strict, ffill, delay.")
    exit_fallback_policy = backtest_settings["BACKTEST_EXIT_FALLBACK_POLICY"]
    if exit_fallback_policy not in {"ffill", "none"}:
        raise SystemExit("backtest.exit_fallback_policy must be one of: ffill, none.")

    execution_model = build_execution_model(
        execution_cfg_resolved,
        default_cost_bps=backtest_settings["BACKTEST_COST_BPS"],
        default_exit_price_policy=exit_price_policy,
        default_exit_fallback_policy=exit_fallback_policy,
        default_price_col=price_col,
    )
    cost_bps_effective = backtest_settings["BACKTEST_COST_BPS"]
    cost_bps_report = None
    if isinstance(execution_model.cost_model, BpsCostModel):
        cost_bps_effective = float(execution_model.cost_model.bps)
        cost_bps_report = cost_bps_effective

    default_sim_liquidity_col = str(
        getattr(execution_model.slippage_model, "amount_col", "medadv20_amount")
        or "medadv20_amount"
    )
    default_sim_portfolio_value = float(
        getattr(execution_model.slippage_model, "portfolio_value", 1_000_000.0) or 1_000_000.0
    )
    try:
        execution_sim_config = build_execution_sim_config(
            backtest_cfg.get("execution_sim"),
            default_portfolio_value=default_sim_portfolio_value,
            default_liquidity_col=default_sim_liquidity_col,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    execution_sim_pricing_cols = required_execution_sim_columns(
        execution_sim_config,
        price_col=execution_model.entry_policy.price_col,
        tradable_col=backtest_settings["BACKTEST_TRADABLE_COL"],
    )
    execution_pricing_cols = required_pricing_columns(execution_model) | execution_sim_pricing_cols
    return {
        "BACKTEST_EXIT_PRICE_POLICY": execution_model.exit_policy.price_policy,
        "BACKTEST_EXIT_FALLBACK_POLICY": execution_model.exit_policy.fallback_policy,
        "execution_model": execution_model,
        "execution_sim_config": execution_sim_config,
        "EXECUTION_SIM_PRICING_COLS": execution_sim_pricing_cols,
        "EXECUTION_PRICING_COLS": execution_pricing_cols,
        "BACKTEST_COST_BPS_EFFECTIVE": cost_bps_effective,
        "BACKTEST_COST_BPS_REPORT": cost_bps_report,
        "BACKTEST_EXECUTION_SOURCE": execution_source,
    }


__all__ = ["resolve_execution_settings"]
