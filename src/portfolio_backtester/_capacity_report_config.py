"""Capacity report: thresholds, metric helpers, and report payload construction."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .capacity_report_support import mapping

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
