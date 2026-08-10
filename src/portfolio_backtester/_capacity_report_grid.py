"""Capacity report: grid row construction and top-level report builder."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from ._capacity_concentration import concentration_by_group
from ._capacity_report_config import (
    THRESHOLD_PROFILES,
    CapacityThresholds,
    _build_report_payload,
    _capacity_limits,
    _evaluate_row,
    _grid_row_from_results,
    _prepare_grid_config,
    _primary_participation_rate,
    _ratio,
    _top_unfilled_orders,
)
from .capacity_report_support import (
    build_execution_context,
    execution_sim_raw,
    normalize_positions_frame,
    normalize_pricing_frame,
    read_frame,
    read_yaml_mapping,
    write_csv,
)
from .execution_sim import (
    ExecutionAdjustedNavResult,
    build_execution_sim_config,
    required_execution_sim_columns,
    simulate_execution_adjusted_nav,
    simulate_ideal_daily_nav,
)
from .liquidity_proxy import _derive_execution_liquidity_proxy_columns


def _build_grid_row(
    *,
    positions: pd.DataFrame,
    pricing: pd.DataFrame,
    sim_raw: Mapping[str, Any],
    portfolio_value: float,
    participation_rate: float,
    liquidity_cols: list[str] | None,
    execution_context: Mapping[str, Any],
    thresholds: CapacityThresholds,
) -> tuple[dict[str, Any], list[dict[str, Any]], ExecutionAdjustedNavResult]:
    grid_raw = _prepare_grid_config(
        sim_raw=sim_raw,
        portfolio_value=portfolio_value,
        participation_rate=participation_rate,
        liquidity_cols=liquidity_cols,
    )
    sim_config = build_execution_sim_config(
        grid_raw,
        default_portfolio_value=portfolio_value,
        default_liquidity_col=str(execution_context["default_liquidity_col"]),
    )
    required_columns = required_execution_sim_columns(
        sim_config,
        price_col=str(execution_context["price_col"]),
        tradable_col=execution_context.get("tradable_col"),
    )
    pricing_for_sim = _derive_execution_liquidity_proxy_columns(pricing.copy(), required_columns)
    missing = sorted(col for col in required_columns if col not in pricing_for_sim.columns)
    if missing:
        raise SystemExit("Pricing panel is missing capacity columns: " + ", ".join(missing))

    ideal = simulate_ideal_daily_nav(
        positions,
        pricing_for_sim,
        price_col=str(execution_context["price_col"]),
        limit_up_col=_optional_rule_col(pricing_for_sim, "limit_up"),
        limit_down_col=_optional_rule_col(pricing_for_sim, "limit_down"),
        listing_status_col=_optional_rule_col(pricing_for_sim, "listing_status"),
        transaction_cost_bps=float(execution_context["transaction_cost_bps"]),
        trading_days_per_year=int(execution_context["trading_days_per_year"]),
        portfolio_value=float(portfolio_value),
        trade_fee_model=execution_context.get("trade_fee_model"),
    )
    executed = simulate_execution_adjusted_nav(
        positions,
        pricing_for_sim,
        sim_config,
        price_col=str(execution_context["price_col"]),
        tradable_col=execution_context.get("tradable_col")
        if execution_context.get("tradable_col") in pricing_for_sim.columns
        else None,
        buy_tradable_col=(
            "is_buy_tradable" if "is_buy_tradable" in pricing_for_sim.columns else None
        ),
        sell_tradable_col=(
            "is_sell_tradable" if "is_sell_tradable" in pricing_for_sim.columns else None
        ),
        limit_up_col=_optional_rule_col(pricing_for_sim, "limit_up"),
        limit_down_col=_optional_rule_col(pricing_for_sim, "limit_down"),
        listing_status_col=_optional_rule_col(pricing_for_sim, "listing_status"),
        transaction_cost_bps=float(execution_context["transaction_cost_bps"]),
        trading_days_per_year=int(execution_context["trading_days_per_year"]),
        trade_fee_model=execution_context.get("trade_fee_model"),
    )
    row = _grid_row_from_results(
        ideal=ideal,
        executed=executed,
        portfolio_value=portfolio_value,
        participation_rate=participation_rate,
    )
    row["return_degradation"] = (
        row["ideal_total_return"] - row["exec_total_return"]
        if row["ideal_total_return"] is not None and row["exec_total_return"] is not None
        else None
    )
    row["sharpe_degradation"] = (
        row["ideal_sharpe"] - row["exec_sharpe"]
        if row["ideal_sharpe"] is not None and row["exec_sharpe"] is not None
        else None
    )
    row["return_retention"] = _ratio(row["exec_total_return"], row["ideal_total_return"])
    row["sharpe_retention"] = _ratio(row["exec_sharpe"], row["ideal_sharpe"])
    failed = _evaluate_row(row, thresholds)
    row["passed"] = not failed
    row["binding_constraints"] = ",".join(failed)
    return row, _top_unfilled_orders(executed.orders), executed


def _optional_rule_col(pricing: pd.DataFrame, default_name: str) -> str | None:
    """Phase 4: resolve an optional market-rule column name from ``pricing``."""
    return default_name if default_name in pricing.columns else None


def build_capacity_report(
    *,
    run_dir: Path,
    config_path: Path,
    positions_path: Path,
    pricing_path: Path,
    portfolio_values: list[float],
    participation_rates: list[float],
    liquidity_cols: list[str] | None,
    threshold_profile: str,
    primary_participation_rate: float | None,
    output_csv: Path | None,
    market_override: str | None = None,
    industry_col: str | None = None,
) -> dict[str, Any]:
    config = read_yaml_mapping(config_path)
    thresholds = THRESHOLD_PROFILES[threshold_profile]
    positions = normalize_positions_frame(read_frame(positions_path))
    pricing = normalize_pricing_frame(read_frame(pricing_path))
    execution_context = build_execution_context(config)
    sim_raw = execution_sim_raw(config)
    rows: list[dict[str, Any]] = []
    examples_by_key: dict[tuple[float, float], list[dict[str, Any]]] = {}
    executed_by_key: dict[tuple[float, float], Any] = {}
    resolved_industry_col = industry_col or execution_context.get("industry_col")
    for portfolio_value in portfolio_values:
        for participation_rate in participation_rates:
            row, examples, executed = _build_grid_row(
                positions=positions,
                pricing=pricing,
                sim_raw=sim_raw,
                portfolio_value=portfolio_value,
                participation_rate=participation_rate,
                liquidity_cols=liquidity_cols,
                execution_context=execution_context,
                thresholds=thresholds,
            )
            rows.append(row)
            key = (float(portfolio_value), float(participation_rate))
            examples_by_key[key] = examples
            executed_by_key[key] = executed
    primary = _primary_participation_rate(
        configured=primary_participation_rate,
        grid=participation_rates,
    )
    first_failing = _capacity_limits(rows, primary_participation_rate=primary)["first_failing_grid"]
    binding_examples: list[dict[str, Any]] = []
    if first_failing is not None:
        key = (
            float(first_failing["portfolio_value"]),
            float(first_failing["participation_rate"]),
        )
        binding_examples = examples_by_key.get(key, [])
    # Phase 5: concentration of executed notional, computed on the primary-rate grid.
    primary_executed = None
    for (_pv, pr), executed in executed_by_key.items():
        if abs(pr - primary) < 1e-12:
            primary_executed = executed
            break
    concentration = None
    if primary_executed is not None:
        concentration = concentration_by_group(
            positions=positions,
            pricing=pricing,
            executed=primary_executed,
            liquidity_col=execution_context.get("default_liquidity_col"),
            industry_col=resolved_industry_col,
        )
    if output_csv is not None:
        write_csv(rows, output_csv)
    market = market_override or str(config.get("market", "unknown")).strip() or "unknown"
    return _build_report_payload(
        rows=rows,
        binding_examples=binding_examples,
        thresholds=thresholds,
        threshold_profile=threshold_profile,
        primary_participation_rate=primary,
        positions=positions,
        pricing=pricing,
        run_dir=run_dir,
        config_path=config_path,
        positions_path=positions_path,
        pricing_path=pricing_path,
        output_csv=output_csv,
        market=market,
        concentration=concentration,
        config=config,
    )
