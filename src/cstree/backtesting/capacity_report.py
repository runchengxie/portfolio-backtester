from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..liquidity_proxy import _derive_execution_liquidity_proxy_columns
from .capacity_report_support import (
    build_execution_context,
    capacity_cfg,
    coerce_liquidity_cols,
    execution_sim_raw,
    float_grid,
    json_default,
    mapping,
    normalize_positions_frame,
    normalize_pricing_frame,
    parse_csv_floats,
    read_frame,
    read_json_mapping,
    read_yaml_mapping,
    resolve_path,
    resolve_positions_path,
    resolve_pricing_path,
    write_csv,
)
from .execution_sim import (
    build_execution_sim_config,
    required_execution_sim_columns,
    simulate_execution_adjusted_nav,
    simulate_ideal_daily_nav,
)

DEFAULT_PORTFOLIO_VALUES = (
    500_000.0,
    1_000_000.0,
    2_000_000.0,
    5_000_000.0,
    10_000_000.0,
    50_000_000.0,
    100_000_000.0,
)
DEFAULT_PARTICIPATION_RATES = (0.01, 0.03, 0.05, 0.10)
DEFAULT_PRIMARY_PARTICIPATION_RATE = 0.05


@dataclass(frozen=True)
class CapacityThresholds:
    min_fill_ratio: float
    max_avg_cash_weight: float
    max_final_cash_weight: float
    min_sharpe_retention: float
    min_return_retention: float
    max_abandoned_buy_order_rate: float | None = None
    max_delayed_sell_order_rate: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_fill_ratio": self.min_fill_ratio,
            "max_avg_cash_weight": self.max_avg_cash_weight,
            "max_final_cash_weight": self.max_final_cash_weight,
            "min_sharpe_retention": self.min_sharpe_retention,
            "min_return_retention": self.min_return_retention,
            "max_abandoned_buy_order_rate": self.max_abandoned_buy_order_rate,
            "max_delayed_sell_order_rate": self.max_delayed_sell_order_rate,
        }


THRESHOLD_PROFILES = {
    "neutral": CapacityThresholds(
        min_fill_ratio=0.95,
        max_avg_cash_weight=0.05,
        max_final_cash_weight=0.10,
        min_sharpe_retention=0.70,
        min_return_retention=0.60,
        max_abandoned_buy_order_rate=0.05,
        max_delayed_sell_order_rate=0.05,
    ),
    "conservative": CapacityThresholds(
        min_fill_ratio=0.98,
        max_avg_cash_weight=0.03,
        max_final_cash_weight=0.05,
        min_sharpe_retention=0.80,
        min_return_retention=0.75,
        max_abandoned_buy_order_rate=0.02,
        max_delayed_sell_order_rate=0.02,
    ),
}


def _prepare_grid_config(
    *,
    sim_raw: Mapping[str, Any],
    portfolio_value: float,
    participation_rate: float,
    liquidity_cols: list[str] | None,
) -> dict[str, Any]:
    cfg = dict(sim_raw)
    cfg.update(
        {
            "enabled": True,
            "portfolio_value": float(portfolio_value),
            "participation_rate": float(participation_rate),
        }
    )
    if liquidity_cols:
        cfg["liquidity_cols"] = list(liquidity_cols)
    return cfg


def _finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _ratio(numerator: object, denominator: object) -> float | None:
    top = _finite(numerator)
    bottom = _finite(denominator)
    if top is None or bottom is None or bottom <= 0:
        return None
    return float(top / bottom)


def _cash_constraint_metric(
    row: Mapping[str, Any],
    *,
    shortfall_key: str,
    fallback_key: str,
) -> tuple[str, float | None]:
    shortfall = _finite(row.get(shortfall_key))
    if shortfall is not None:
        return shortfall_key, shortfall
    return fallback_key, _finite(row.get(fallback_key))


def _count_order_statuses(orders: pd.DataFrame) -> dict[str, Any]:
    if orders.empty:
        return {
            "orders": 0,
            "abandoned_buy_orders": 0,
            "abandoned_buy_order_rate": 0.0,
            "delayed_sell_orders": 0,
            "delayed_sell_order_rate": 0.0,
        }
    buy_orders = orders[orders["side"].astype(str).str.lower() == "buy"]
    sell_orders = orders[orders["side"].astype(str).str.lower() == "sell"]
    abandoned = int((buy_orders["status"] == "abandoned_zero_fill").sum())
    delayed = int((sell_orders["status"] == "delayed_sell").sum())
    return {
        "orders": int(orders.shape[0]),
        "abandoned_buy_orders": abandoned,
        "abandoned_buy_order_rate": abandoned / int(buy_orders.shape[0])
        if not buy_orders.empty
        else 0.0,
        "delayed_sell_orders": delayed,
        "delayed_sell_order_rate": delayed / int(sell_orders.shape[0])
        if not sell_orders.empty
        else 0.0,
    }


def _participation_quantiles(fills: pd.DataFrame, participation_rate: float) -> dict[str, Any]:
    if fills.empty or "capacity_notional" not in fills.columns:
        return {
            "p95_participation": None,
            "p99_participation": None,
            "p95_capacity_utilization": None,
        }
    capacity = pd.to_numeric(fills["capacity_notional"], errors="coerce")
    filled = pd.to_numeric(fills["filled_notional"], errors="coerce")
    utilization = (filled / capacity.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    participation = utilization * float(participation_rate)
    return {
        "p95_participation": _finite(participation.quantile(0.95)),
        "p99_participation": _finite(participation.quantile(0.99)),
        "p95_capacity_utilization": _finite(utilization.quantile(0.95)),
    }


def _top_unfilled_orders(orders: pd.DataFrame, *, limit: int = 10) -> list[dict[str, Any]]:
    if orders.empty or "unfilled_notional" not in orders.columns:
        return []
    work = orders.copy()
    work["unfilled_notional"] = pd.to_numeric(work["unfilled_notional"], errors="coerce")
    work = work[work["unfilled_notional"] > 0].sort_values("unfilled_notional", ascending=False)
    columns = [
        "rebalance_date",
        "entry_date",
        "side",
        "symbol",
        "requested_notional",
        "filled_notional",
        "unfilled_notional",
        "status",
        "fill_days",
    ]
    return work[[col for col in columns if col in work.columns]].head(limit).to_dict("records")


def _evaluate_row(row: Mapping[str, Any], thresholds: CapacityThresholds) -> list[str]:
    failed: list[str] = []
    fill_ratio = _finite(row.get("fill_ratio"))
    if fill_ratio is None or fill_ratio < thresholds.min_fill_ratio:
        failed.append("fill_ratio")
    avg_cash_name, avg_cash = _cash_constraint_metric(
        row,
        shortfall_key="avg_execution_shortfall_cash_weight",
        fallback_key="avg_cash_weight",
    )
    if avg_cash is None or avg_cash > thresholds.max_avg_cash_weight:
        failed.append(avg_cash_name)
    final_cash_name, final_cash = _cash_constraint_metric(
        row,
        shortfall_key="final_execution_shortfall_cash_weight",
        fallback_key="final_cash_weight",
    )
    if final_cash is None or final_cash > thresholds.max_final_cash_weight:
        failed.append(final_cash_name)
    sharpe_retention = _finite(row.get("sharpe_retention"))
    if sharpe_retention is not None and sharpe_retention < thresholds.min_sharpe_retention:
        failed.append("sharpe_retention")
    return_retention = _finite(row.get("return_retention"))
    if return_retention is not None and return_retention < thresholds.min_return_retention:
        failed.append("return_retention")
    abandoned_rate = _finite(row.get("abandoned_buy_order_rate"))
    if (
        thresholds.max_abandoned_buy_order_rate is not None
        and abandoned_rate is not None
        and abandoned_rate > thresholds.max_abandoned_buy_order_rate
    ):
        failed.append("abandoned_buy_order_rate")
    delayed_rate = _finite(row.get("delayed_sell_order_rate"))
    if (
        thresholds.max_delayed_sell_order_rate is not None
        and delayed_rate is not None
        and delayed_rate > thresholds.max_delayed_sell_order_rate
    ):
        failed.append("delayed_sell_order_rate")
    return failed


def _grid_row_from_results(
    *,
    ideal: Any,
    executed: Any,
    portfolio_value: float,
    participation_rate: float,
) -> dict[str, Any]:
    ideal_stats = mapping(ideal.summary.get("stats"))
    exec_stats = mapping(executed.summary.get("stats"))
    return {
        "portfolio_value": float(portfolio_value),
        "participation_rate": float(participation_rate),
        "status": executed.summary.get("status"),
        "ideal_status": ideal.summary.get("status"),
        "ideal_total_return": _finite(ideal_stats.get("total_return")),
        "exec_total_return": _finite(exec_stats.get("total_return")),
        "ideal_sharpe": _finite(ideal_stats.get("sharpe")),
        "exec_sharpe": _finite(exec_stats.get("sharpe")),
        "ideal_max_drawdown": _finite(ideal_stats.get("max_drawdown")),
        "exec_max_drawdown": _finite(exec_stats.get("max_drawdown")),
        "fill_ratio": _finite(executed.summary.get("fill_ratio")),
        "buy_fill_ratio": _finite(executed.summary.get("buy_fill_ratio")),
        "sell_fill_ratio": _finite(executed.summary.get("sell_fill_ratio")),
        "unfilled_notional": _finite(executed.summary.get("unfilled_notional")),
        "avg_cash_weight": _finite(executed.summary.get("avg_cash_weight")),
        "avg_target_cash_weight": _finite(executed.summary.get("avg_target_cash_weight")),
        "avg_execution_shortfall_cash_weight": _finite(
            executed.summary.get("avg_execution_shortfall_cash_weight")
        ),
        "final_cash_weight": _finite(executed.summary.get("final_cash_weight")),
        "final_target_cash_weight": _finite(executed.summary.get("final_target_cash_weight")),
        "final_execution_shortfall_cash_weight": _finite(
            executed.summary.get("final_execution_shortfall_cash_weight")
        ),
        "daily_rows": int(executed.summary.get("daily_rows") or 0),
        **_count_order_statuses(executed.orders),
        **_participation_quantiles(executed.fills, participation_rate),
    }


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
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
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
    return row, _top_unfilled_orders(executed.orders)


def _primary_participation_rate(*, configured: object, grid: list[float]) -> float:
    desired = _finite(configured)
    if desired is None:
        desired = DEFAULT_PRIMARY_PARTICIPATION_RATE
    return min(grid, key=lambda value: abs(value - float(desired)))


def _capacity_limits(
    rows: list[dict[str, Any]],
    *,
    primary_participation_rate: float,
) -> dict[str, Any]:
    primary_rows = sorted(
        [
            row
            for row in rows
            if abs(float(row["participation_rate"]) - float(primary_participation_rate)) < 1e-12
        ],
        key=lambda row: float(row["portfolio_value"]),
    )
    passing = [row for row in primary_rows if bool(row.get("passed"))]
    recommended = float(passing[-1]["portfolio_value"]) if passing else None
    first_failing = None
    if recommended is not None:
        first_failing = next(
            (
                row
                for row in primary_rows
                if float(row["portfolio_value"]) > recommended and not bool(row.get("passed"))
            ),
            None,
        )
    elif primary_rows:
        first_failing = primary_rows[0]
    hard_capacity = (
        float(first_failing["portfolio_value"])
        if first_failing is not None
        else (float(primary_rows[-1]["portfolio_value"]) if primary_rows else None)
    )
    constraints: list[str] = []
    if first_failing is not None:
        constraints = [
            item for item in str(first_failing.get("binding_constraints") or "").split(",") if item
        ]
    return {
        "recommended_capacity": recommended,
        "hard_capacity": hard_capacity,
        "binding_constraints": constraints,
        "first_failing_grid": first_failing,
    }


def _date_text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _build_report_payload(
    *,
    rows: list[dict[str, Any]],
    binding_examples: list[dict[str, Any]],
    thresholds: CapacityThresholds,
    threshold_profile: str,
    primary_participation_rate: float,
    positions: pd.DataFrame,
    pricing: pd.DataFrame,
    run_dir: Path,
    config_path: Path,
    positions_path: Path,
    pricing_path: Path,
    output_csv: Path | None,
    market: str,
) -> dict[str, Any]:
    limits = _capacity_limits(rows, primary_participation_rate=primary_participation_rate)
    return {
        "schema": "a_share.capacity.v1" if market == "a_share" else "capacity.v1",
        "status": "passed" if limits["recommended_capacity"] is not None else "failed",
        "market": market,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "input_files": {
            "config": str(config_path),
            "positions": str(positions_path),
            "pricing": str(pricing_path),
        },
        "output_files": {"capacity_grid_csv": str(output_csv) if output_csv else None},
        "data_window": {
            "pricing_start": _date_text(pricing["trade_date"].min()),
            "pricing_end": _date_text(pricing["trade_date"].max()),
            "rebalance_start": _date_text(positions["rebalance_date"].min()),
            "rebalance_end": _date_text(positions["rebalance_date"].max()),
            "pricing_rows": int(pricing.shape[0]),
            "position_rows": int(positions.shape[0]),
            "rebalances": int(positions["rebalance_date"].nunique()),
            "symbols": int(pricing["symbol"].nunique()),
        },
        "portfolio_grid": sorted({float(row["portfolio_value"]) for row in rows}),
        "participation_rate_grid": sorted({float(row["participation_rate"]) for row in rows}),
        "participation_rate_assumption": float(primary_participation_rate),
        "threshold_profile": threshold_profile,
        "thresholds": thresholds.to_dict(),
        "recommended_capacity": limits["recommended_capacity"],
        "hard_capacity": limits["hard_capacity"],
        "binding_constraints": limits["binding_constraints"],
        "first_failing_grid": limits["first_failing_grid"],
        "binding_examples": binding_examples,
        "metrics_by_grid": rows,
        "limitations": [
            "Daily ADV capacity report; does not model intraday queue priority, VWAP/TWAP timing, "
            "auction mechanics, or broker fills.",
            "Cash thresholds use execution_shortfall_cash_weight when available, so intentional "
            "target cash from sub-100% gross overlays is reported but not treated as a "
            "fill failure.",
            "Return and Sharpe retention checks are skipped when the ideal metric is non-positive.",
        ],
    }


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
) -> dict[str, Any]:
    config = read_yaml_mapping(config_path)
    thresholds = THRESHOLD_PROFILES[threshold_profile]
    positions = normalize_positions_frame(read_frame(positions_path))
    pricing = normalize_pricing_frame(read_frame(pricing_path))
    execution_context = build_execution_context(config)
    sim_raw = execution_sim_raw(config)
    rows: list[dict[str, Any]] = []
    examples_by_key: dict[tuple[float, float], list[dict[str, Any]]] = {}
    for portfolio_value in portfolio_values:
        for participation_rate in participation_rates:
            row, examples = _build_grid_row(
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
            examples_by_key[(float(portfolio_value), float(participation_rate))] = examples
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
    )


def add_capacity_report_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--run-dir", required=True, help="Existing cstree run directory.")
    parser.add_argument(
        "--config",
        default=None,
        help="Pipeline config to use. Defaults to <run-dir>/config.used.yml.",
    )
    parser.add_argument(
        "--positions-file",
        default=None,
        help="Override positions_by_rebalance file.",
    )
    parser.add_argument(
        "--pricing-file",
        default=None,
        help="Pricing panel with trade_date, symbol, price, and liquidity columns.",
    )
    parser.add_argument(
        "--portfolio-value",
        action="append",
        default=None,
        help="Portfolio value grid. Repeat or pass comma-separated values.",
    )
    parser.add_argument(
        "--participation-rate",
        action="append",
        default=None,
        help="Daily participation-rate grid. Repeat or pass comma-separated values.",
    )
    parser.add_argument(
        "--liquidity-col",
        action="append",
        default=None,
        help="Liquidity column used for capacity. Repeat for min-of-columns behavior.",
    )
    parser.add_argument(
        "--primary-participation-rate",
        type=float,
        default=None,
        help="Participation-rate assumption used for recommended/hard capacity.",
    )
    parser.add_argument(
        "--threshold-profile",
        choices=sorted(THRESHOLD_PROFILES),
        default="neutral",
        help="Capacity pass/fail threshold profile.",
    )
    parser.add_argument("--output-dir", default=None, help="Directory for capacity outputs.")
    parser.add_argument("--output-csv", default=None, help="Capacity grid CSV path.")
    parser.add_argument("--output-json", default=None, help="Capacity report JSON path.")
    parser.add_argument("--market", default=None, help="Override market label in the report.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging level.",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(levelname)s: %(message)s",
    )
    run_dir = resolve_path(args.run_dir)
    if run_dir is None or not run_dir.exists():
        raise SystemExit(f"Run directory not found: {args.run_dir}")
    config_path = (
        resolve_path(args.config, base_dir=run_dir) if args.config else run_dir / "config.used.yml"
    )
    if config_path is None or not config_path.exists():
        raise SystemExit(f"Config file not found: {config_path}")
    config = read_yaml_mapping(config_path)
    cfg = capacity_cfg(config)
    summary_path = run_dir / "summary.json"
    summary = read_json_mapping(summary_path) if summary_path.exists() else {}
    sim_raw = execution_sim_raw(config)
    positions_path = resolve_positions_path(run_dir=run_dir, summary=summary, args=args, cfg=cfg)
    pricing_path = resolve_pricing_path(run_dir=run_dir, summary=summary, args=args, cfg=cfg)
    portfolio_values = float_grid(
        cli_values=parse_csv_floats(args.portfolio_value),
        cfg_values=cfg.get("portfolio_values")
        or sim_raw.get("portfolio_values")
        or sim_raw.get("portfolio_value"),
        fallback=DEFAULT_PORTFOLIO_VALUES,
        label="portfolio_values",
    )
    participation_rates = float_grid(
        cli_values=parse_csv_floats(args.participation_rate),
        cfg_values=cfg.get("participation_rates")
        or sim_raw.get("participation_rates")
        or sim_raw.get("participation_rate"),
        fallback=DEFAULT_PARTICIPATION_RATES,
        label="participation_rates",
    )
    liquidity_cols = coerce_liquidity_cols(args=args, cfg=cfg, sim_raw=sim_raw)
    output_dir = resolve_path(args.output_dir or cfg.get("output_dir"), base_dir=run_dir)
    output_dir = output_dir or run_dir
    output_csv = resolve_path(args.output_csv or cfg.get("output_csv"), base_dir=output_dir)
    output_json = resolve_path(args.output_json or cfg.get("output_json"), base_dir=output_dir)
    output_csv = output_csv or output_dir / "capacity_grid.csv"
    output_json = output_json or output_dir / "capacity_report.json"
    payload = build_capacity_report(
        run_dir=run_dir,
        config_path=config_path,
        positions_path=positions_path,
        pricing_path=pricing_path,
        portfolio_values=portfolio_values,
        participation_rates=participation_rates,
        liquidity_cols=liquidity_cols,
        threshold_profile=args.threshold_profile,
        primary_participation_rate=args.primary_participation_rate
        if args.primary_participation_rate is not None
        else cfg.get("primary_participation_rate"),
        output_csv=output_csv,
        market_override=args.market,
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, default=json_default),
        encoding="utf-8",
    )
    logging.info("Capacity grid CSV written to %s", output_csv)
    logging.info("Capacity report JSON written to %s", output_json)
    return payload
