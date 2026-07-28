"""Capacity report: command-line argument parsing and CLI entry point."""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any

from ._capacity_report_config import (
    DEFAULT_PARTICIPATION_RATES,
    DEFAULT_PORTFOLIO_VALUES,
    THRESHOLD_PROFILES,
)
from ._capacity_report_grid import build_capacity_report
from .capacity_report_support import (
    capacity_cfg,
    coerce_liquidity_cols,
    execution_sim_raw,
    float_grid,
    json_default,
    parse_csv_floats,
    read_json_mapping,
    read_yaml_mapping,
    resolve_path,
    resolve_positions_path,
    resolve_pricing_path,
)


def add_capacity_report_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Existing pipeline run directory created by strategy-pipeline.",
    )
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
