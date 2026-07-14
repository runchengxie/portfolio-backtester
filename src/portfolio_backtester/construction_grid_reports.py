from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

FIELDNAMES = [
    "variant",
    "scored_file",
    "summary_path",
    "target_col",
    "price_col",
    "eval_signal_col",
    "backtest_signal_col",
    "top_k",
    "rank_offset",
    "short_k",
    "long_only",
    "cost_bps",
    "buffer_exit",
    "buffer_entry",
    "weighting",
    "weighting_liquidity_col",
    "liquidity_floor_col",
    "liquidity_floor_quantile",
    "max_turnover_per_rebalance",
    "score_postprocess_method",
    "score_postprocess_columns",
    "dynamic_ensemble_active",
    "dynamic_ensemble_signal_cols",
    "dynamic_ensemble_avg_active_factor_count",
    "dynamic_ensemble_avg_factor_turnover",
    "dynamic_ensemble_avg_stock_turnover",
    "factor_correlation_threshold",
    "risk_penalty_columns",
    "risk_penalty_strength",
    "eval_ic_mean",
    "eval_ic_ir",
    "eval_long_short",
    "eval_turnover_mean",
    "backtest_periods",
    "backtest_total_return",
    "backtest_gross_total_return",
    "backtest_ann_return",
    "backtest_ann_vol",
    "backtest_sharpe",
    "backtest_max_drawdown",
    "backtest_avg_turnover",
    "backtest_avg_cost_drag",
    "active_total_return",
    "information_ratio",
    "tracking_error",
    "beta",
    "alpha",
    "corr",
    "benchmark_name",
    "benchmark_returns_file",
    "exposure_available",
    "status",
    "error",
]


def select_construction_variant_with_inertia(
    rows: list[dict[str, Any]],
    *,
    previous_variant: str | None = None,
    objective_col: str = "information_ratio",
    switch_penalty: float = 0.0,
    min_improvement: float = 0.0,
) -> dict[str, Any]:
    candidates = [
        row
        for row in rows
        if row.get("status") == "ok" and _finite_float(row.get(objective_col)) is not None
    ]
    if not candidates:
        return {
            "status": "no_valid_candidates",
            "selected_variant": previous_variant,
            "previous_variant": previous_variant,
            "objective_col": objective_col,
            "switch_penalty": float(switch_penalty),
            "min_improvement": float(min_improvement),
            "best_variant": None,
            "best_objective": None,
            "previous_objective": None,
            "improvement": None,
            "switched": False,
        }
    ranked = sorted(candidates, key=lambda row: float(row[objective_col]), reverse=True)
    best = ranked[0]
    best_objective = float(best[objective_col])
    previous = next((row for row in candidates if row.get("variant") == previous_variant), None)
    if previous is None or previous_variant is None:
        return {
            "status": "selected",
            "selected_variant": best.get("variant"),
            "previous_variant": previous_variant,
            "objective_col": objective_col,
            "switch_penalty": float(switch_penalty),
            "min_improvement": float(min_improvement),
            "best_variant": best.get("variant"),
            "best_objective": best_objective,
            "previous_objective": None,
            "improvement": None,
            "switched": bool(previous_variant and best.get("variant") != previous_variant),
        }
    previous_objective = float(previous[objective_col])
    improvement = best_objective - previous_objective
    required_improvement = float(min_improvement) + float(switch_penalty)
    if best.get("variant") != previous_variant and improvement <= required_improvement:
        selected = previous
        switched = False
    else:
        selected = best
        switched = best.get("variant") != previous_variant
    return {
        "status": "selected",
        "selected_variant": selected.get("variant"),
        "previous_variant": previous_variant,
        "objective_col": objective_col,
        "switch_penalty": float(switch_penalty),
        "min_improvement": float(min_improvement),
        "best_variant": best.get("variant"),
        "best_objective": best_objective,
        "previous_objective": previous_objective,
        "improvement": improvement,
        "switched": bool(switched),
    }


def build_inertia_selection_report(
    rows: list[dict[str, Any]],
    selection_cfg: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(selection_cfg, dict):
        raise SystemExit("construction_grid.rolling_selection must be a mapping.")
    return {
        "schema_version": 1,
        "artifact_type": "portfolio_backtester.construction_grid_rolling_selection",
        **select_construction_variant_with_inertia(
            rows,
            previous_variant=selection_cfg.get("previous_variant"),
            objective_col=str(selection_cfg.get("objective_col") or "information_ratio"),
            switch_penalty=float(selection_cfg.get("switch_penalty", 0.0)),
            min_improvement=float(selection_cfg.get("min_improvement", 0.0)),
        ),
    }


def _finite_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def write_reports(
    rows: list[dict[str, Any]],
    *,
    output_csv: Path | None,
    output_json: Path | None,
) -> None:
    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(rows, ensure_ascii=True, indent=2, default=str),
            encoding="utf-8",
        )
