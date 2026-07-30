"""Order construction submodules (split from orders.py for maintainability)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..execution import DetailedTradeFeeModel
from .capacity import (
    _capacity_notional,
    _capacity_weight,
    _execution_window_dates,
    _price_at,
)
from .config import (
    ExecutionSimConfig,
)
from .models import (
    _ExecutionTables,
    _NavOrder,
    _OrderSink,
    _trade_fee,
)
from .reporting import (
    _format_date,
)

TradeFeeModel = DetailedTradeFeeModel


def _execute_sell_orders(
    *,
    rebalance_date: pd.Timestamp,
    entry_date: pd.Timestamp,
    next_entry_date: pd.Timestamp | None,
    requests: dict[str, float],
    current_weights: dict[str, float],
    cash_weight: float,
    config: ExecutionSimConfig,
    tables: _ExecutionTables,
    sink: _OrderSink,
) -> float:
    remaining = dict(requests)
    states = _build_order_states(requests)
    window_dates = _execution_window_dates(
        entry_date,
        max_days=config.sell_max_days,
        next_entry_date=next_entry_date,
        trade_dates=tables.trade_dates,
        date_to_idx=tables.date_to_idx,
    )
    for day_number, trade_date in enumerate(window_dates, start=1):
        for symbol in sorted(remaining):
            before = remaining[symbol]
            capacity = _capacity_weight(
                symbol,
                trade_date,
                config=config,
                price_table=tables.price_table,
                tradable_table=tables.sell_tradable_table,
                liquidity_tables=tables.liquidity_tables,
            )
            fill = min(before, capacity)
            if fill > 1e-12:
                remaining[symbol] = max(before - fill, 0.0)
                current_weights[symbol] = max(current_weights.get(symbol, 0.0) - fill, 0.0)
                if current_weights[symbol] <= 1e-12:
                    current_weights.pop(symbol, None)
                cash_weight += fill
                _record_fill(
                    sink.fill_rows,
                    rebalance_date=rebalance_date,
                    entry_date=entry_date,
                    trade_date=trade_date,
                    day_number=day_number,
                    side="sell",
                    symbol=symbol,
                    remaining_before=before,
                    capacity=capacity,
                    fill=fill,
                    config=config,
                )
                _update_state(states[symbol], trade_date, fill)
            if remaining.get(symbol, 0.0) <= 1e-12:
                remaining.pop(symbol, None)
        if not remaining:
            break
    _append_order_rows(
        sink.order_rows,
        rebalance_date=rebalance_date,
        entry_date=entry_date,
        side="sell",
        requests=requests,
        remaining=remaining,
        states=states,
        max_days=len(window_dates),
        config=config,
        unfilled_status="delayed_sell",
    )
    return cash_weight




def _execute_buy_orders(
    *,
    rebalance_date: pd.Timestamp,
    entry_date: pd.Timestamp,
    requests: dict[str, float],
    current_weights: dict[str, float],
    cash_weight: float,
    config: ExecutionSimConfig,
    tables: _ExecutionTables,
    sink: _OrderSink,
) -> float:
    remaining = dict(requests)
    states = _build_order_states(requests)
    abandoned: set[str] = set()
    window_dates = _execution_window_dates(
        entry_date,
        max_days=config.buy_max_days,
        next_entry_date=None,
        trade_dates=tables.trade_dates,
        date_to_idx=tables.date_to_idx,
    )
    for day_number, trade_date in enumerate(window_dates, start=1):
        daily_fills: dict[str, tuple[float, float]] = {}
        for symbol in sorted(remaining):
            if symbol in abandoned:
                continue
            before = remaining[symbol]
            capacity = _capacity_weight(
                symbol,
                trade_date,
                config=config,
                price_table=tables.price_table,
                tradable_table=tables.buy_tradable_table,
                liquidity_tables=tables.liquidity_tables,
            )
            fill = min(before, capacity)
            daily_fills[symbol] = (capacity, fill)

        total_requested_fill = sum(fill for _, fill in daily_fills.values())
        scale = 1.0
        if total_requested_fill > max(cash_weight, 0.0) and total_requested_fill > 0:
            scale = max(cash_weight, 0.0) / total_requested_fill

        for symbol in sorted(remaining):
            if symbol in abandoned:
                continue
            before = remaining[symbol]
            capacity, raw_fill = daily_fills.get(symbol, (0.0, 0.0))
            fill = min(before, raw_fill * scale)
            if fill > 1e-12:
                remaining[symbol] = max(before - fill, 0.0)
                current_weights[symbol] = current_weights.get(symbol, 0.0) + fill
                cash_weight = max(cash_weight - fill, 0.0)
                _record_fill(
                    sink.fill_rows,
                    rebalance_date=rebalance_date,
                    entry_date=entry_date,
                    trade_date=trade_date,
                    day_number=day_number,
                    side="buy",
                    symbol=symbol,
                    remaining_before=before,
                    capacity=capacity,
                    fill=fill,
                    config=config,
                )
                _update_state(states[symbol], trade_date, fill)
                states[symbol]["zero_fill_days"] = 0
            else:
                if capacity <= 1e-12:
                    states[symbol]["zero_fill_days"] += 1
                    if (
                        config.zero_fill_abort_days_buy is not None
                        and states[symbol]["zero_fill_days"] >= config.zero_fill_abort_days_buy
                    ):
                        abandoned.add(symbol)
            if remaining.get(symbol, 0.0) <= 1e-12:
                remaining.pop(symbol, None)
        if not remaining:
            break
        if set(remaining).issubset(abandoned):
            break

    _append_order_rows(
        sink.order_rows,
        rebalance_date=rebalance_date,
        entry_date=entry_date,
        side="buy",
        requests=requests,
        remaining=remaining,
        states=states,
        max_days=len(window_dates),
        config=config,
        unfilled_status="cancelled_buy_deadline",
        abandoned=abandoned,
    )
    return cash_weight




def _execute_nav_orders_for_day(
    *,
    open_orders: list[_NavOrder],
    shares: dict[str, float],
    cash_ref: dict[str, float],
    trade_date: pd.Timestamp,
    trade_idx: int,
    tables: _ExecutionTables,
    config: ExecutionSimConfig,
    cost_rate: float,
    trade_fee_model: TradeFeeModel | None,
    fill_rows: list[dict[str, Any]],
) -> tuple[float, float]:
    traded_notional = 0.0
    transaction_cost = 0.0
    sell_traded, sell_cost = _execute_nav_sell_orders_for_day(
        open_orders=open_orders,
        shares=shares,
        cash_ref=cash_ref,
        trade_date=trade_date,
        trade_idx=trade_idx,
        tables=tables,
        config=config,
        cost_rate=cost_rate,
        trade_fee_model=trade_fee_model,
        fill_rows=fill_rows,
    )
    traded_notional += sell_traded
    transaction_cost += sell_cost

    buy_traded, buy_cost = _execute_nav_buy_orders_for_day(
        open_orders=open_orders,
        shares=shares,
        cash_ref=cash_ref,
        trade_date=trade_date,
        trade_idx=trade_idx,
        tables=tables,
        config=config,
        cost_rate=cost_rate,
        trade_fee_model=trade_fee_model,
        fill_rows=fill_rows,
    )
    traded_notional += buy_traded
    transaction_cost += buy_cost
    return float(traded_notional), float(transaction_cost)




def _execute_nav_sell_orders_for_day(
    *,
    open_orders: list[_NavOrder],
    shares: dict[str, float],
    cash_ref: dict[str, float],
    trade_date: pd.Timestamp,
    trade_idx: int,
    tables: _ExecutionTables,
    config: ExecutionSimConfig,
    cost_rate: float,
    trade_fee_model: TradeFeeModel | None,
    fill_rows: list[dict[str, Any]],
) -> tuple[float, float]:
    traded_notional = 0.0
    transaction_cost = 0.0
    for order in sorted(
        [
            item
            for item in open_orders
            if item.side == "sell" and not _nav_order_is_complete(item)
        ],
        key=lambda item: item.symbol,
    ):
        price = _price_at(order.symbol, trade_date, tables.price_table)
        if not np.isfinite(price):
            continue
        held_quantity = max(float(shares.get(order.symbol, 0.0)), 0.0)
        capacity = _capacity_notional(
            order.symbol,
            trade_date,
            config=config,
            price_table=tables.price_table,
            tradable_table=tables.sell_tradable_table,
            liquidity_tables=tables.liquidity_tables,
        )
        remaining_quantity = (
            max(float(order.remaining_quantity), 0.0)
            if order.remaining_quantity is not None
            else max(float(order.remaining_notional) / float(price), 0.0)
        )
        fill_quantity = min(
            remaining_quantity,
            held_quantity,
            max(float(capacity) / float(price), 0.0),
        )
        fill = fill_quantity * float(price)
        if fill <= 1e-8:
            continue
        remaining_before_notional = remaining_quantity * float(price)
        shares[order.symbol] = max(held_quantity - fill_quantity, 0.0)
        if shares[order.symbol] <= 1e-10:
            shares.pop(order.symbol, None)
        cost = _trade_fee(fill, side="sell", cost_rate=cost_rate, fee_model=trade_fee_model)
        cash_ref["cash"] = float(cash_ref.get("cash", 0.0)) + fill - cost
        _update_nav_order(
            order,
            trade_date,
            fill,
            filled_quantity=fill_quantity,
        )
        _record_nav_fill(
            fill_rows,
            order=order,
            trade_date=trade_date,
            trade_idx=trade_idx,
            capacity_notional=capacity,
            filled_notional=fill,
            transaction_cost=cost,
            remaining_before_notional=remaining_before_notional,
        )
        traded_notional += fill
        transaction_cost += cost
    return float(traded_notional), float(transaction_cost)




def _execute_nav_buy_orders_for_day(
    *,
    open_orders: list[_NavOrder],
    shares: dict[str, float],
    cash_ref: dict[str, float],
    trade_date: pd.Timestamp,
    trade_idx: int,
    tables: _ExecutionTables,
    config: ExecutionSimConfig,
    cost_rate: float,
    trade_fee_model: TradeFeeModel | None,
    fill_rows: list[dict[str, Any]],
) -> tuple[float, float]:
    candidates = [
        item for item in open_orders if item.side == "buy" and item.remaining_notional > 1e-8
    ]
    raw_fills: dict[str, tuple[_NavOrder, float, float, float]] = {}
    for order in sorted(candidates, key=lambda item: item.symbol):
        price = _price_at(order.symbol, trade_date, tables.price_table)
        capacity = _capacity_notional(
            order.symbol,
            trade_date,
            config=config,
            price_table=tables.price_table,
            tradable_table=tables.buy_tradable_table,
            liquidity_tables=tables.liquidity_tables,
        )
        raw_fill = min(float(order.remaining_notional), capacity)
        if raw_fill <= 1e-8:
            if capacity <= 1e-8:
                order.zero_fill_days += 1
            continue
        raw_fills[order.symbol] = (order, float(price), capacity, raw_fill)

    total_raw_fill = sum(item[3] for item in raw_fills.values())
    if total_raw_fill <= 1e-8:
        return 0.0, 0.0
    cash = max(float(cash_ref.get("cash", 0.0)), 0.0)
    total_cash_required = total_raw_fill + sum(
        _trade_fee(item[3], side="buy", cost_rate=cost_rate, fee_model=trade_fee_model)
        for item in raw_fills.values()
    )
    scale = min(1.0, cash / total_cash_required) if total_cash_required > 0 else 0.0
    if scale <= 1e-12:
        return 0.0, 0.0

    traded_notional = 0.0
    transaction_cost = 0.0
    for symbol, (order, price, capacity, raw_fill) in sorted(raw_fills.items()):
        del symbol
        fill = raw_fill * scale
        if fill <= 1e-8:
            continue
        cost = _trade_fee(fill, side="buy", cost_rate=cost_rate, fee_model=trade_fee_model)
        quantity = fill / float(price)
        shares[order.symbol] = float(shares.get(order.symbol, 0.0)) + quantity
        cash_ref["cash"] = float(cash_ref.get("cash", 0.0)) - fill - cost
        _update_nav_order(order, trade_date, fill)
        order.zero_fill_days = 0
        _record_nav_fill(
            fill_rows,
            order=order,
            trade_date=trade_date,
            trade_idx=trade_idx,
            capacity_notional=capacity,
            filled_notional=fill,
            transaction_cost=cost,
        )
        traded_notional += fill
        transaction_cost += cost
    return float(traded_notional), float(transaction_cost)




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



