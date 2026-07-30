"""NAV order state aggregation, completion checks and fill recording (split from orders_nav.py)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import ExecutionSimConfig
from .models import _NavOrder
from .reporting import _format_date


def _nav_order_is_complete(order: _NavOrder) -> bool:
    if order.remaining_quantity is not None:
        return bool(order.remaining_quantity <= 1e-10)
    return bool(order.remaining_notional <= 1e-8)


def _update_nav_order(
    order: _NavOrder,
    trade_date: pd.Timestamp,
    fill: float,
    *,
    filled_quantity: float | None = None,
) -> None:
    order.filled_notional += float(fill)
    if (
        filled_quantity is not None
        and order.requested_quantity is not None
        and order.remaining_quantity is not None
    ):
        order.filled_quantity += float(filled_quantity)
        order.remaining_quantity = max(
            float(order.remaining_quantity) - float(filled_quantity),
            0.0,
        )
        reference_price = (
            float(order.requested_notional) / float(order.requested_quantity)
            if order.requested_quantity > 0
            else 0.0
        )
        order.remaining_notional = float(order.remaining_quantity) * reference_price
    else:
        order.remaining_notional = max(
            float(order.remaining_notional) - float(fill),
            0.0,
        )
    if order.first_fill_date is None:
        order.first_fill_date = trade_date
    order.last_fill_date = trade_date
    order.fill_days += 1


def _record_nav_fill(
    fill_rows: list[dict[str, Any]],
    *,
    order: _NavOrder,
    trade_date: pd.Timestamp,
    trade_idx: int,
    capacity_notional: float,
    filled_notional: float,
    transaction_cost: float,
    remaining_before_notional: float | None = None,
) -> None:
    fill_rows.append(
        {
            "rebalance_date": _format_date(order.rebalance_date),
            "entry_date": _format_date(order.entry_date),
            "trade_date": _format_date(trade_date),
            "day_number": int(trade_idx - order.start_idx + 1),
            "side": order.side,
            "symbol": order.symbol,
            "remaining_before_notional": float(
                order.remaining_notional + filled_notional
                if remaining_before_notional is None
                else remaining_before_notional
            ),
            "capacity_notional": float(capacity_notional),
            "filled_notional": float(filled_notional),
            "transaction_cost": float(transaction_cost),
        }
    )


def _nav_order_should_abort_buy(order: _NavOrder, config: ExecutionSimConfig) -> bool:
    return config.zero_fill_abort_days_buy is not None and order.zero_fill_days >= int(
        config.zero_fill_abort_days_buy
    )


def _finalize_open_nav_orders(
    open_orders: list[_NavOrder],
    order_rows: list[dict[str, Any]],
    *,
    trade_date: pd.Timestamp,
    participation_rate: float,
    status_by_side: dict[str, str],
) -> None:
    for order in open_orders:
        if _nav_order_is_complete(order):
            order.status = "filled"
        else:
            order.status = status_by_side.get(order.side, "cancelled")
        _append_nav_order_row(
            order_rows,
            order,
            trade_date=trade_date,
            participation_rate=participation_rate,
        )


def _append_nav_order_row(
    order_rows: list[dict[str, Any]],
    order: _NavOrder,
    *,
    trade_date: pd.Timestamp,
    participation_rate: float,
) -> None:
    status = order.status or ("filled" if _nav_order_is_complete(order) else "open")
    fill_ratio = (
        float(order.filled_quantity / order.requested_quantity)
        if order.requested_quantity is not None and order.requested_quantity > 0
        else (
            float(order.filled_notional / order.requested_notional)
            if order.requested_notional > 0
            else np.nan
        )
    )
    order_rows.append(
        {
            "rebalance_date": _format_date(order.rebalance_date),
            "entry_date": _format_date(order.entry_date),
            "side": order.side,
            "symbol": order.symbol,
            "requested_notional": float(order.requested_notional),
            "filled_notional": float(order.filled_notional),
            "unfilled_notional": float(max(order.remaining_notional, 0.0)),
            "fill_ratio": fill_ratio,
            "status": status,
            "first_fill_date": _format_date(order.first_fill_date),
            "last_fill_date": _format_date(order.last_fill_date),
            "closed_date": _format_date(trade_date),
            "fill_days": int(order.fill_days),
            "max_days": int(order.max_days),
            "zero_fill_days": int(order.zero_fill_days),
            "participation_rate": float(participation_rate),
        }
    )


def _build_order_states(requests: dict[str, float]) -> dict[str, dict[str, Any]]:
    return {
        symbol: {
            "requested": float(amount),
            "filled": 0.0,
            "first_fill_date": None,
            "last_fill_date": None,
            "fill_days": 0,
            "zero_fill_days": 0,
        }
        for symbol, amount in requests.items()
    }


def _update_state(state: dict[str, Any], trade_date: pd.Timestamp, fill: float) -> None:
    state["filled"] += float(fill)
    if state["first_fill_date"] is None:
        state["first_fill_date"] = trade_date
    state["last_fill_date"] = trade_date
    state["fill_days"] += 1


def _append_order_rows(
    order_rows: list[dict[str, Any]],
    *,
    rebalance_date: pd.Timestamp,
    entry_date: pd.Timestamp,
    side: str,
    requests: dict[str, float],
    remaining: dict[str, float],
    states: dict[str, dict[str, Any]],
    max_days: int,
    config: ExecutionSimConfig,
    unfilled_status: str,
    abandoned: set[str] | None = None,
) -> None:
    abandoned = abandoned or set()
    for symbol in sorted(requests):
        state = states[symbol]
        requested = float(requests[symbol])
        filled = min(float(state["filled"]), requested)
        unfilled = max(float(remaining.get(symbol, 0.0)), 0.0)
        if unfilled <= 1e-12:
            status = "filled"
        elif symbol in abandoned:
            status = "abandoned_zero_fill"
        else:
            status = unfilled_status
        order_rows.append(
            {
                "rebalance_date": _format_date(rebalance_date),
                "entry_date": _format_date(entry_date),
                "side": side,
                "symbol": symbol,
                "requested_weight": requested,
                "filled_weight": filled,
                "unfilled_weight": unfilled,
                "requested_notional": requested * config.portfolio_value,
                "filled_notional": filled * config.portfolio_value,
                "unfilled_notional": unfilled * config.portfolio_value,
                "fill_ratio": filled / requested if requested > 0 else np.nan,
                "status": status,
                "first_fill_date": _format_date(state["first_fill_date"]),
                "last_fill_date": _format_date(state["last_fill_date"]),
                "fill_days": int(state["fill_days"]),
                "max_days": int(max_days),
                "zero_fill_days": int(state["zero_fill_days"]),
                "participation_rate": float(config.participation_rate),
            }
        )


def _record_fill(
    fill_rows: list[dict[str, Any]],
    *,
    rebalance_date: pd.Timestamp,
    entry_date: pd.Timestamp,
    trade_date: pd.Timestamp,
    day_number: int,
    side: str,
    symbol: str,
    remaining_before: float,
    capacity: float,
    fill: float,
    config: ExecutionSimConfig,
) -> None:
    fill_rows.append(
        {
            "rebalance_date": _format_date(rebalance_date),
            "entry_date": _format_date(entry_date),
            "trade_date": _format_date(trade_date),
            "day_number": int(day_number),
            "side": side,
            "symbol": symbol,
            "remaining_before_weight": float(remaining_before),
            "capacity_weight": float(capacity),
            "filled_weight": float(fill),
            "capacity_notional": float(capacity) * config.portfolio_value,
            "filled_notional": float(fill) * config.portfolio_value,
        }
    )
