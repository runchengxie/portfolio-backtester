"""Stable execution summaries for research applications."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd

from .staggered_cohort_execution_state import StaggeredCohortExecutionResult

EXECUTION_SUMMARY_SCHEMA = "portfolio.execution_summary.v1"


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return cast(pd.Series, pd.to_numeric(frame[column], errors="coerce"))


def summarize_staggered_execution(
    result: StaggeredCohortExecutionResult,
    *,
    variant: str,
    horizon: int,
    single_side_cost_bps: float,
) -> dict[str, Any]:
    """Return a portfolio-owned summary without exposing ledger implementation types."""

    daily = result.daily.sort_values("trade_date", kind="mergesort").reset_index(drop=True)
    if daily.empty:
        raise ValueError("staggered execution result has no daily rows")
    net = _numeric(daily, "net_return")
    if net.isna().any() or not np.isfinite(net.to_numpy(dtype=float)).all():
        raise ValueError("staggered execution net returns must be complete and finite")
    wealth = (1.0 + net).cumprod()
    total_return = float(wealth.iloc[-1] - 1.0)
    annualized_return = float((1.0 + total_return) ** (252.0 / len(net)) - 1.0)
    volatility = float(net.std(ddof=1) * np.sqrt(252.0)) if len(net) > 1 else np.nan
    sharpe = (
        float(net.mean() / net.std(ddof=1) * np.sqrt(252.0))
        if len(net) > 1 and net.std(ddof=1) > 0
        else np.nan
    )
    drawdown = wealth / wealth.cummax() - 1.0
    gross_nav = _numeric(daily, "gross_nav_before_cost")
    traded_ratio = _numeric(daily, "traded_notional") / gross_nav.replace(0.0, np.nan)
    summary = result.summary
    return {
        "schema_version": EXECUTION_SUMMARY_SCHEMA,
        "variant": str(variant),
        "horizon": int(horizon),
        "single_side_cost_bps": float(single_side_cost_bps),
        "sessions": len(daily),
        "total_return": total_return,
        "annualized_return": annualized_return,
        "mean_daily_net_return": float(net.mean()),
        "annualized_volatility": volatility,
        "annualized_sharpe_zero_rf": sharpe,
        "max_drawdown": float(drawdown.min()),
        "total_transaction_cost": float(_numeric(daily, "transaction_cost").sum()),
        "total_traded_notional": float(_numeric(daily, "traded_notional").sum()),
        "mean_traded_notional_to_nav": float(traded_ratio.mean()),
        "mean_cash_weight": float(_numeric(daily, "cash_weight").mean()),
        "mean_blocked_buy_weight": float(_numeric(daily, "blocked_buy_weight").mean()),
        "mean_blocked_sell_weight": float(_numeric(daily, "blocked_sell_weight").mean()),
        "terminal_complete": bool(summary.get("complete_nav")),
        "final_nav": summary.get("final_nav"),
        "diagnostic_mark_to_open_nav": summary.get("diagnostic_mark_to_open_nav"),
        "terminal_unclosed_position_count": int(
            summary.get("terminal_unclosed_position_count") or 0
        ),
        "tradability_price": summary.get("tradability_price"),
        "valuation_and_fill_price": summary.get("valuation_and_fill_price"),
    }


def execution_summary_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Validate and deterministically order a grid of public execution summaries."""

    frame = pd.DataFrame(rows)
    required = {
        "schema_version",
        "variant",
        "horizon",
        "single_side_cost_bps",
        "total_return",
        "mean_daily_net_return",
        "mean_traded_notional_to_nav",
        "terminal_complete",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"execution summaries are missing columns: {missing}")
    if set(frame["schema_version"]) != {EXECUTION_SUMMARY_SCHEMA}:
        raise ValueError("execution summaries use an unsupported schema")
    if frame.duplicated(["variant", "horizon", "single_side_cost_bps"]).any():
        raise ValueError("execution summaries contain duplicate grid cells")
    return frame.sort_values(
        ["horizon", "single_side_cost_bps", "variant"], kind="mergesort"
    ).reset_index(drop=True)


__all__ = [
    "EXECUTION_SUMMARY_SCHEMA",
    "execution_summary_frame",
    "summarize_staggered_execution",
]
