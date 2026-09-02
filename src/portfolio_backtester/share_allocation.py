"""Deterministic share allocation from portfolio targets and market references."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class ShareAllocationResult:
    """Round-lot-aware allocation result for a fixed reference snapshot."""

    allocations: pd.DataFrame
    investable_cash: float
    est_total: float
    cash_left: float


def allocate_equal_weight_shares(
    selection: pd.DataFrame,
    *,
    cash: float,
    buffer_bps: float = 0.0,
) -> ShareAllocationResult:
    """Allocate equal target value across rows, rounded down to whole lots.

    `selection` must contain `symbol`, `price`, and `round_lot`. Optional
    `order_book_id`, `side`, and `rank` metadata are preserved in the result.
    Market-price discovery and trading-calendar semantics are deliberately
    outside this function.
    """

    if selection.empty:
        raise ValueError("No holdings selected for allocation.")

    required = {"symbol", "price", "round_lot"}
    missing = sorted(required.difference(selection.columns))
    if missing:
        raise ValueError("Allocation input is missing required column(s): " + ", ".join(missing))

    investable_cash = float(cash) * max(0.0, 1.0 - float(buffer_bps) / 10_000.0)
    target_value = investable_cash / float(len(selection))

    rows: list[dict[str, object]] = []
    for _, row in selection.iterrows():
        symbol = str(row["symbol"])
        price = float(row["price"])
        round_lot = int(row["round_lot"])
        if price <= 0:
            raise ValueError(f"price must be positive for {symbol}: {price}")
        if round_lot <= 0:
            raise ValueError(f"round_lot must be positive for {symbol}: {round_lot}")

        lot_cost = price * round_lot
        lots = math.floor(target_value / lot_cost)
        shares = lots * round_lot
        est_value = shares * price

        rank_value = row.get("rank")
        rank = None
        if rank_value is not None and not pd.isna(rank_value):
            rank = int(rank_value)

        rows.append(
            {
                "symbol": symbol,
                "order_book_id": str(row.get("order_book_id", symbol)),
                "side": str(row.get("side", "long")),
                "rank": rank,
                "price": price,
                "round_lot": round_lot,
                "target_value": target_value,
                "lot_cost": lot_cost,
                "lots": lots,
                "shares": shares,
                "est_value": est_value,
                "gap_to_target": target_value - est_value,
            }
        )

    allocations = pd.DataFrame(rows)
    est_total = float(allocations["est_value"].sum())
    cash_left = investable_cash - est_total
    return ShareAllocationResult(
        allocations=allocations,
        investable_cash=investable_cash,
        est_total=est_total,
        cash_left=cash_left,
    )
