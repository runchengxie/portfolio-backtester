from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import pandas as pd

__all__ = [
    "ExecutionAdjustedNavResult",
    "ExecutionSimResult",
    "UnifiedLedger",
    "to_unified_ledger",
]


@dataclass(frozen=True)
class ExecutionSimResult:
    summary: dict[str, Any]
    orders: pd.DataFrame
    fills: pd.DataFrame

    def to_unified_ledger(
        self,
        *,
        portfolio_value: float | None = None,
    ) -> UnifiedLedger:
        return to_unified_ledger(self, portfolio_value=portfolio_value)


@dataclass(frozen=True)
class ExecutionAdjustedNavResult:
    summary: dict[str, Any]
    daily: pd.DataFrame
    orders: pd.DataFrame
    fills: pd.DataFrame

    def to_unified_ledger(
        self,
        *,
        portfolio_value: float | None = None,
    ) -> UnifiedLedger:
        return to_unified_ledger(self, portfolio_value=portfolio_value)


@dataclass(frozen=True)
class UnifiedLedger:
    """Roadmap-defined 8-field reconciled ledger.

    Produced by :func:`to_unified_ledger` so the independent execution-sim
    engine can feed the shared daily-ledger contract without changing the
    historical ``not_available`` semantics of callers that do not opt in.
    """

    targets: pd.DataFrame
    orders: pd.DataFrame
    fills: pd.DataFrame
    daily_positions: pd.DataFrame
    daily_cash: pd.DataFrame
    daily_nav: pd.DataFrame
    cost_breakdown: pd.DataFrame
    turnover_breakdown: pd.DataFrame


def to_unified_ledger(
    result: Any,
    *,
    portfolio_value: float | None = None,
) -> UnifiedLedger:
    """Adapt an :class:`ExecutionSimResult` / :class:`ExecutionAdjustedNavResult`.

    Maps the engine's ``daily`` / ``orders`` / ``fills`` frames onto the eight
    reconciled ledger fields:

    - ``targets``: per-rebalance target weights, rebuilt from order requests.
    - ``orders`` / ``fills``: the engine's order and fill frames, unchanged.
    - ``daily_positions`` / ``daily_cash`` / ``daily_nav``: cash and position
      value series derived from the daily frame.
    - ``cost_breakdown``: transaction-cost aggregation by side.
    - ``turnover_breakdown``: filled-notional aggregation by side.
    """

    orders = _as_frame(getattr(result, "orders", None))
    fills = _as_frame(getattr(result, "fills", None))
    daily = _as_frame(getattr(result, "daily", None))

    resolved_value = _resolve_portfolio_value(result, portfolio_value)

    targets = _build_targets(orders, resolved_value)
    daily_positions = _build_daily_series(daily, "invested_value", "positions_value")
    daily_cash = _build_daily_series(daily, "cash", "cash")
    daily_nav = _build_daily_series(daily, "portfolio_value", "nav")
    cost_breakdown = _build_cost_breakdown(fills, daily)
    turnover_breakdown = _build_turnover_breakdown(orders, fills)

    return UnifiedLedger(
        targets=targets,
        orders=orders,
        fills=fills,
        daily_positions=daily_positions,
        daily_cash=daily_cash,
        daily_nav=daily_nav,
        cost_breakdown=cost_breakdown,
        turnover_breakdown=turnover_breakdown,
    )


def _as_frame(value: Any) -> pd.DataFrame:
    if value is None:
        return pd.DataFrame()
    return cast(pd.DataFrame, value)


def _resolve_portfolio_value(result: Any, portfolio_value: float | None) -> float:
    if portfolio_value is not None:
        return float(portfolio_value)
    summary = getattr(result, "summary", None)
    if isinstance(summary, Mapping):
        config = summary.get("config") if isinstance(summary, Mapping) else None
        if isinstance(config, Mapping) and config.get("portfolio_value"):
            return float(config["portfolio_value"])
        if summary.get("portfolio_value"):
            return float(summary["portfolio_value"])
    return 1_000_000.0


def _build_targets(orders: pd.DataFrame, portfolio_value: float) -> pd.DataFrame:
    if orders.empty:
        return pd.DataFrame(
            columns=[
                "rebalance_date",
                "entry_date",
                "symbol",
                "side",
                "target_weight",
                "target_notional",
            ]
        )
    rows = []
    for _, order in orders.iterrows():
        requested_weight = order.get("requested_weight")
        if requested_weight is None or pd.isna(requested_weight):
            requested_notional = float(order.get("requested_notional", 0.0) or 0.0)
            target_weight = requested_notional / portfolio_value if portfolio_value > 0 else 0.0
        else:
            target_weight = float(requested_weight)
        rows.append(
            {
                "rebalance_date": order.get("rebalance_date"),
                "entry_date": order.get("entry_date"),
                "symbol": order.get("symbol"),
                "side": order.get("side"),
                "target_weight": float(target_weight),
                "target_notional": float(target_weight * portfolio_value),
            }
        )
    return pd.DataFrame(rows)


def _build_daily_series(daily: pd.DataFrame, value_col: str, out_col: str) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame(columns=["trade_date", out_col])
    series = pd.to_numeric(daily.get(value_col), errors="coerce")
    return pd.DataFrame(
        {
            "trade_date": daily["trade_date"].astype(str).tolist(),
            out_col: series.to_numpy(dtype=float),
        }
    )


def _build_cost_breakdown(fills: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    cost_by_side: dict[str, float] = {}
    total_cost = 0.0
    if not fills.empty and "transaction_cost" in fills.columns:
        fills = fills.copy()
        fills["transaction_cost"] = pd.to_numeric(
            fills["transaction_cost"], errors="coerce"
        ).fillna(0.0)
        for side, sub in fills.groupby("side"):
            cost_by_side[str(side)] = float(sub["transaction_cost"].sum())
        total_cost = float(fills["transaction_cost"].sum())
    elif not daily.empty and "transaction_cost" in daily.columns:
        total_cost = float(
            pd.to_numeric(daily["transaction_cost"], errors="coerce").fillna(0.0).sum()
        )
    return pd.DataFrame(
        {
            "side": ["total", *sorted(cost_by_side)],
            "transaction_cost": [
                total_cost,
                *[cost_by_side[s] for s in sorted(cost_by_side)],
            ],
        }
    )


def _build_turnover_breakdown(orders: pd.DataFrame, fills: pd.DataFrame) -> pd.DataFrame:
    notional_by_side: dict[str, float] = {}
    has_fills = not fills.empty and "filled_notional" in fills.columns
    has_orders = not orders.empty and "filled_notional" in orders.columns
    if has_fills:
        fills = fills.copy()
        fills["filled_notional"] = pd.to_numeric(fills["filled_notional"], errors="coerce").fillna(
            0.0
        )
        for side, sub in fills.groupby("side"):
            notional_by_side[str(side)] = float(sub["filled_notional"].sum())
    elif has_orders:
        orders = orders.copy()
        orders["filled_notional"] = pd.to_numeric(
            orders["filled_notional"], errors="coerce"
        ).fillna(0.0)
        for side, sub in orders.groupby("side"):
            notional_by_side[str(side)] = float(sub["filled_notional"].sum())
    total = float(sum(notional_by_side.values()))
    return pd.DataFrame(
        {
            "side": ["total", *sorted(notional_by_side)],
            "filled_notional": [
                total,
                *[notional_by_side[s] for s in sorted(notional_by_side)],
            ],
        }
    )
