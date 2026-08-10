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


# Maps the daily/fill ``cost_*`` columns onto the exact CostBreakdown field
# names used by the ledger contract.
_COST_SUB_COLS = [
    "cost_commission",
    "cost_stamp_tax",
    "cost_transfer_fee",
    "cost_spread",
    "cost_temporary_impact",
    "cost_permanent_impact",
    "cost_opportunity",
    "cost_financing",
]
_COST_FIELD_BY_COL = {
    "cost_commission": "commission",
    "cost_stamp_tax": "stamp_tax",
    "cost_transfer_fee": "transfer_fee",
    "cost_spread": "spread_cost",
    "cost_temporary_impact": "temporary_impact",
    "cost_permanent_impact": "permanent_impact",
    "cost_opportunity": "opportunity_cost",
    "cost_financing": "financing_cost",
}


def _cost_row(sub: pd.DataFrame) -> dict[str, float]:
    row: dict[str, float] = {}
    for col in _COST_SUB_COLS:
        if col in sub.columns:
            row[_COST_FIELD_BY_COL[col]] = float(
                pd.to_numeric(sub[col], errors="coerce").fillna(0.0).sum()
            )
    if "transaction_cost" in sub.columns and not row:
        row["transaction_cost"] = float(
            pd.to_numeric(sub["transaction_cost"], errors="coerce").fillna(0.0).sum()
        )
    return row


def _aggregate_cost_columns(frame: pd.DataFrame, *, by_side: bool) -> dict[str, dict[str, float]]:
    groups: list[tuple[str, pd.DataFrame]] = (
        [(str(s), sub) for s, sub in frame.groupby("side")] if by_side else [("total", frame)]
    )
    out: dict[str, dict[str, float]] = {}
    for label, sub in groups:
        row = _cost_row(sub)
        if row:
            out[label] = row
    return out


def _build_cost_breakdown(fills: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Aggregate stage-3 transaction-cost sub-items by side.

    Prefers the per-fill side split when ``fills`` carries ``cost_commission``
    etc.; otherwise falls back to the daily sub-item columns (summed across the
    whole period). When no sub-item columns exist the legacy single
    ``transaction_cost`` column is aggregated by side so the ledger contract is
    preserved.
    """
    source: dict[str, dict[str, float]] | None = None
    if not fills.empty and (
        "cost_commission" in fills.columns or "transaction_cost" in fills.columns
    ):
        source = _aggregate_cost_columns(fills, by_side=True)
    elif not daily.empty and (
        "cost_commission" in daily.columns or "transaction_cost" in daily.columns
    ):
        source = _aggregate_cost_columns(daily, by_side=False)

    if not source:
        return pd.DataFrame(columns=["side", "transaction_cost"])

    # The "total" row must aggregate every row, not just a single side.
    total_row: dict[str, float] = {}
    for row in source.values():
        for k, v in row.items():
            total_row[k] = total_row.get(k, 0.0) + v
    source = {"total": total_row, **source}

    all_keys: list[str] = []
    for row in source.values():
        for k in row:
            if k not in all_keys:
                all_keys.append(k)

    labels = ["total", *sorted(s for s in source if s != "total")]
    data: dict[str, list[float] | list[str]] = {"side": labels}
    for key in all_keys:
        data[key] = [float(source.get(label, {}).get(key, 0.0)) for label in labels]
    # Derived aggregate columns (CostBreakdown contract): fee_cost / slippage_cost.
    fee_cols = [c for c in ("commission", "stamp_tax", "transfer_fee") if c in data]
    slip_cols = [
        c
        for c in (
            "spread_cost",
            "temporary_impact",
            "permanent_impact",
            "opportunity_cost",
            "financing_cost",
        )
        if c in data
    ]
    if fee_cols:
        data["fee_cost"] = [float(sum(data[c][i] for c in fee_cols)) for i in range(len(labels))]
    if slip_cols:
        data["slippage_cost"] = [
            float(sum(data[c][i] for c in slip_cols)) for i in range(len(labels))
        ]
    # Legacy ``transaction_cost`` alias: aggregate of every sub-item so existing
    # consumers (and conservation assertions) keep working.
    if "transaction_cost" not in data:
        data["transaction_cost"] = [float(sum(source.get(label, {}).values())) for label in labels]
    return pd.DataFrame(data)


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
