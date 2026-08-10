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


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _ratio(numerator: Any, denominator: Any) -> float | None:
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


def _primary_rows_sorted(
    rows: list[dict[str, Any]], *, primary_participation_rate: float
) -> list[dict[str, Any]]:
    """Phase 5: rows at the primary participation rate, sorted by portfolio_value ascending."""
    selected = [
        row
        for row in rows
        if abs(float(row["participation_rate"]) - float(primary_participation_rate)) < 1e-12
    ]
    return sorted(selected, key=lambda row: float(row["portfolio_value"]))


def _max_satisfying_capacity(
    *,
    primary_rows: list[dict[str, Any]],
    metric_key: str,
    threshold: float,
    higher_is_better: bool = True,
) -> float | None:
    """Phase 5: largest portfolio_value whose ``metric_key`` satisfies the threshold.

    Returns ``None`` when no row satisfies it. Uses a single linear interpolation step
    between the last satisfying grid point and the next larger one to refine the estimate
    without running additional simulations.
    """
    satisfying: list[tuple[float, float]] = []
    for row in primary_rows:
        value = _finite(row.get(metric_key))
        if value is None:
            continue
        ok = value >= threshold if higher_is_better else value <= threshold
        if ok:
            satisfying.append((float(row["portfolio_value"]), value))
    if not satisfying:
        return None
    satisfying.sort(key=lambda item: item[0])
    best_value, best_metric = satisfying[-1]
    if len(satisfying) == len(primary_rows):
        # Every grid point satisfies: capacity is at or beyond the largest scanned value.
        return best_value
    # Refine between best_value and the next larger grid point using linear interpolation.
    for row in primary_rows:
        pv = float(row["portfolio_value"])
        if pv <= best_value:
            continue
        metric = _finite(row.get(metric_key))
        if metric is None:
            continue
        if best_metric == metric:
            return best_value
        frac = (threshold - best_metric) / (metric - best_metric)
        frac = max(0.0, min(1.0, frac))
        return best_value + frac * (pv - best_value)
    return best_value


def _break_even_capacity(
    rows: list[dict[str, Any]], *, primary_participation_rate: float
) -> float | None:
    """Phase 5: largest portfolio_value with non-negative executed total return."""
    primary_rows = _primary_rows_sorted(rows, primary_participation_rate=primary_participation_rate)
    return _max_satisfying_capacity(
        primary_rows=primary_rows,
        metric_key="exec_total_return",
        threshold=0.0,
        higher_is_better=True,
    )


def _fill_rate_capacity(
    rows: list[dict[str, Any]], *, primary_participation_rate: float, target: float = 0.95
) -> float | None:
    """Phase 5: largest portfolio_value with fill_ratio >= target."""
    primary_rows = _primary_rows_sorted(rows, primary_participation_rate=primary_participation_rate)
    return _max_satisfying_capacity(
        primary_rows=primary_rows,
        metric_key="fill_ratio",
        threshold=float(target),
        higher_is_better=True,
    )


def _alpha_retention_capacity(
    rows: list[dict[str, Any]],
    *,
    primary_participation_rate: float,
    target: float = 0.90,
    metric_key: str = "return_retention",
) -> float | None:
    """Phase 5: largest portfolio_value retaining at least ``target`` of ideal metric."""
    primary_rows = _primary_rows_sorted(rows, primary_participation_rate=primary_participation_rate)
    return _max_satisfying_capacity(
        primary_rows=primary_rows,
        metric_key=metric_key,
        threshold=float(target),
        higher_is_better=True,
    )


def _marginal_impact(
    rows: list[dict[str, Any]], *, primary_participation_rate: float
) -> dict[str, float | None]:
    """Phase 5: marginal impact of one extra unit of capital on return/sharpe retention.

    Estimates the slope of ``exec_total_return`` and ``sharpe_retention`` against
    ``portfolio_value`` over the largest adjacent grid interval (where capacity stress is
    highest). Negative slope means adding capital erodes performance.
    """
    primary_rows = _primary_rows_sorted(rows, primary_participation_rate=primary_participation_rate)
    _none_marginal = {
        "marginal_return_per_unit_capital": None,
        "marginal_sharpe_retention_per_unit_capital": None,
    }
    if len(primary_rows) < 2:
        return _none_marginal
    last = primary_rows[-1]
    prev = primary_rows[-2]
    dpv = float(last["portfolio_value"]) - float(prev["portfolio_value"])
    if dpv <= 0:
        return _none_marginal
    ret_last = _finite(last.get("exec_total_return"))
    ret_prev = _finite(prev.get("exec_total_return"))
    sharpe_last = _finite(last.get("sharpe_retention"))
    sharpe_prev = _finite(prev.get("sharpe_retention"))
    marginal_return = (
        (ret_last - ret_prev) / dpv if ret_last is not None and ret_prev is not None else None
    )
    marginal_sharpe = (
        (sharpe_last - sharpe_prev) / dpv
        if sharpe_last is not None and sharpe_prev is not None
        else None
    )
    return {
        "marginal_return_per_unit_capital": marginal_return,
        "marginal_sharpe_retention_per_unit_capital": marginal_sharpe,
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
    concentration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    limits = _capacity_limits(rows, primary_participation_rate=primary_participation_rate)
    calibration = {
        "break_even_capacity": _break_even_capacity(
            rows, primary_participation_rate=primary_participation_rate
        ),
        "fill_rate_95_capacity": _fill_rate_capacity(
            rows, primary_participation_rate=primary_participation_rate, target=0.95
        ),
        "alpha_retention_90_capacity": _alpha_retention_capacity(
            rows,
            primary_participation_rate=primary_participation_rate,
            target=0.90,
            metric_key="return_retention",
        ),
        "sharpe_retention_90_capacity": _alpha_retention_capacity(
            rows,
            primary_participation_rate=primary_participation_rate,
            target=0.90,
            metric_key="sharpe_retention",
        ),
        **_marginal_impact(rows, primary_participation_rate=primary_participation_rate),
    }
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
        "capacity_calibration": calibration,
        "concentration": concentration,
        "limitations": [
            "Daily ADV capacity report; does not model intraday queue priority, VWAP/TWAP timing, "
            "auction mechanics, or broker fills.",
            "Cash thresholds use execution_shortfall_cash_weight when available, so intentional "
            "target cash from sub-100% gross overlays is reported but not treated as a "
            "fill failure.",
            "Return and Sharpe retention checks are skipped when the ideal metric is non-positive.",
        ],
    }
