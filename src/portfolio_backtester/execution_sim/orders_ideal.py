"""Order construction submodules (split from orders.py for maintainability)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from ..execution import DetailedTradeFeeModel
from ..types import CostBreakdown
from .capacity import (
    _position_values_by_symbol,
    _price_at,
)
from .config import (
    SELL_UNTIL_NEXT_REBALANCE,
    ExecutionSimConfig,
)
from .models import (
    _add_breakdown,
    _ExecutionTables,
    _NavOrder,
    _trade_fee,
)
from .orders_nav import (
    _append_nav_order_row,
    _record_nav_fill_audit,
    _update_nav_order,
)
from .orders_targets import (
    _cost_adjusted_target_notional,
)

TradeFeeModel = DetailedTradeFeeModel


def _rebalance_ideal_target(
    *,
    rebalance_date: pd.Timestamp,
    entry_date: pd.Timestamp,
    target_weights: dict[str, float],
    shares: dict[str, float],
    cash_ref: dict[str, float],
    nav: float,
    trade_idx: int,
    tables: _ExecutionTables,
    config: ExecutionSimConfig,
    last_prices: dict[str, float],
    cost_rate: float,
    trade_fee_model: TradeFeeModel | None,
    order_rows: list[dict[str, Any]],
    fill_rows: list[dict[str, Any]],
) -> tuple[float, CostBreakdown]:
    current_values = _position_values_by_symbol(
        shares,
        entry_date,
        tables.price_table,
        last_prices,
    )
    target_notional = _cost_adjusted_target_notional(
        current_values=current_values,
        target_weights=target_weights,
        nav=nav,
        cost_rate=cost_rate,
    )
    sell_orders, buy_orders = _build_ideal_rebalance_orders(
        rebalance_date=rebalance_date,
        entry_date=entry_date,
        current_values=current_values,
        target_notional=target_notional,
        trade_idx=trade_idx,
    )
    sell_traded, sell_cost = _execute_ideal_sell_orders(
        sell_orders=sell_orders,
        shares=shares,
        cash_ref=cash_ref,
        entry_date=entry_date,
        trade_idx=trade_idx,
        tables=tables,
        config=config,
        last_prices=last_prices,
        cost_rate=cost_rate,
        trade_fee_model=trade_fee_model,
        order_rows=order_rows,
        fill_rows=fill_rows,
    )
    buy_traded, buy_cost = _execute_ideal_buy_orders(
        buy_orders=buy_orders,
        shares=shares,
        cash_ref=cash_ref,
        entry_date=entry_date,
        trade_idx=trade_idx,
        tables=tables,
        config=config,
        last_prices=last_prices,
        cost_rate=cost_rate,
        trade_fee_model=trade_fee_model,
        order_rows=order_rows,
        fill_rows=fill_rows,
    )
    return float(sell_traded + buy_traded), _add_breakdown(sell_cost, buy_cost)


def _build_ideal_rebalance_orders(
    *,
    rebalance_date: pd.Timestamp,
    entry_date: pd.Timestamp,
    current_values: Mapping[str, float],
    target_notional: Mapping[str, float],
    trade_idx: int,
) -> tuple[list[_NavOrder], list[_NavOrder]]:
    sell_orders: list[_NavOrder] = []
    buy_orders: list[_NavOrder] = []
    for symbol in sorted(set(current_values) | set(target_notional)):
        current_notional = float(current_values.get(symbol, 0.0))
        desired_notional = float(target_notional.get(symbol, 0.0))
        delta = desired_notional - current_notional
        if delta < -1e-8:
            sell_orders.append(
                _ideal_nav_order(
                    rebalance_date=rebalance_date,
                    entry_date=entry_date,
                    side="sell",
                    symbol=symbol,
                    notional=abs(float(delta)),
                    trade_idx=trade_idx,
                )
            )
        elif delta > 1e-8:
            buy_orders.append(
                _ideal_nav_order(
                    rebalance_date=rebalance_date,
                    entry_date=entry_date,
                    side="buy",
                    symbol=symbol,
                    notional=float(delta),
                    trade_idx=trade_idx,
                )
            )
    return sell_orders, buy_orders


def _ideal_nav_order(
    *,
    rebalance_date: pd.Timestamp,
    entry_date: pd.Timestamp,
    side: str,
    symbol: str,
    notional: float,
    trade_idx: int,
) -> _NavOrder:
    return _NavOrder(
        rebalance_date=rebalance_date,
        entry_date=entry_date,
        side=side,
        symbol=symbol,
        requested_notional=float(notional),
        remaining_notional=float(notional),
        start_idx=trade_idx,
        max_days=1,
    )


def _build_nav_orders_for_target(
    *,
    rebalance_date: pd.Timestamp,
    entry_date: pd.Timestamp,
    next_entry_date: pd.Timestamp | None,
    target_weights: dict[str, float],
    shares: dict[str, float],
    cash: float,
    nav: float,
    trade_idx: int,
    tables: _ExecutionTables,
    config: ExecutionSimConfig,
    last_prices: dict[str, float],
) -> list[_NavOrder]:
    del cash
    current_values = _position_values_by_symbol(
        shares,
        entry_date,
        tables.price_table,
        last_prices,
    )
    sell_max_days = _nav_sell_max_days(
        config,
        trade_idx=trade_idx,
        next_entry_date=next_entry_date,
        tables=tables,
    )
    orders: list[_NavOrder] = []
    for symbol in sorted(set(current_values) | set(target_weights)):
        current_notional = float(current_values.get(symbol, 0.0))
        target_notional = max(float(target_weights.get(symbol, 0.0)), 0.0) * float(nav)
        delta = target_notional - current_notional
        if delta > 1e-8:
            orders.append(
                _NavOrder(
                    rebalance_date=rebalance_date,
                    entry_date=entry_date,
                    side="buy",
                    symbol=symbol,
                    requested_notional=float(delta),
                    remaining_notional=float(delta),
                    start_idx=trade_idx,
                    max_days=int(config.buy_max_days),
                )
            )
        elif delta < -1e-8:
            amount = abs(float(delta))
            held_quantity = max(float(shares.get(symbol, 0.0)), 0.0)
            reference_price = current_notional / held_quantity if held_quantity > 1e-12 else np.nan
            requested_quantity = (
                amount / reference_price
                if np.isfinite(reference_price) and reference_price > 0
                else None
            )
            orders.append(
                _NavOrder(
                    rebalance_date=rebalance_date,
                    entry_date=entry_date,
                    side="sell",
                    symbol=symbol,
                    requested_notional=amount,
                    remaining_notional=amount,
                    start_idx=trade_idx,
                    max_days=sell_max_days,
                    requested_quantity=requested_quantity,
                    remaining_quantity=requested_quantity,
                )
            )
    return orders


def _nav_sell_max_days(
    config: ExecutionSimConfig,
    *,
    trade_idx: int,
    next_entry_date: pd.Timestamp | None,
    tables: _ExecutionTables,
) -> int:
    if config.sell_max_days == SELL_UNTIL_NEXT_REBALANCE:
        if next_entry_date is not None and next_entry_date in tables.date_to_idx:
            return max(1, int(tables.date_to_idx[next_entry_date] - trade_idx))
        return max(1, int(len(tables.trade_dates) - trade_idx))
    return int(config.sell_max_days)


def _execute_ideal_sell_orders(
    *,
    sell_orders: list[_NavOrder],
    shares: dict[str, float],
    cash_ref: dict[str, float],
    entry_date: pd.Timestamp,
    trade_idx: int,
    tables: _ExecutionTables,
    config: ExecutionSimConfig,
    last_prices: dict[str, float],
    cost_rate: float,
    trade_fee_model: TradeFeeModel | None,
    order_rows: list[dict[str, Any]],
    fill_rows: list[dict[str, Any]],
) -> tuple[float, CostBreakdown]:
    traded_notional = 0.0
    transaction_cost = CostBreakdown()
    for order in sell_orders:
        price = _price_at(order.symbol, entry_date, tables.price_table)
        held_quantity = max(float(shares.get(order.symbol, 0.0)), 0.0)
        held_notional = held_quantity * float(price) if np.isfinite(price) else 0.0
        fill = min(float(order.remaining_notional), held_notional)
        if fill > 1e-8 and np.isfinite(price):
            cost = _apply_ideal_sell_fill(
                order=order,
                shares=shares,
                cash_ref=cash_ref,
                price=float(price),
                fill=fill,
                cost_rate=cost_rate,
                trade_fee_model=trade_fee_model,
                last_prices=last_prices,
                entry_date=entry_date,
                trade_idx=trade_idx,
                fill_rows=fill_rows,
            )
            traded_notional += fill
            transaction_cost = _add_breakdown(transaction_cost, cost)
        order.status = _ideal_sell_status(order, price)
        _append_nav_order_row(
            order_rows,
            order,
            trade_date=entry_date,
            participation_rate=config.participation_rate,
        )
    return float(traded_notional), transaction_cost


def _apply_ideal_sell_fill(
    *,
    order: _NavOrder,
    shares: dict[str, float],
    cash_ref: dict[str, float],
    price: float,
    fill: float,
    cost_rate: float,
    trade_fee_model: TradeFeeModel | None,
    last_prices: dict[str, float],
    entry_date: pd.Timestamp,
    trade_idx: int,
    fill_rows: list[dict[str, Any]],
) -> CostBreakdown:
    held_quantity = max(float(shares.get(order.symbol, 0.0)), 0.0)
    shares[order.symbol] = max(held_quantity - fill / price, 0.0)
    if shares[order.symbol] <= 1e-10:
        shares.pop(order.symbol, None)
    cost = _trade_fee(fill, side="sell", cost_rate=cost_rate, fee_model=trade_fee_model)
    cash_ref["cash"] = float(cash_ref.get("cash", 0.0)) + fill - cost.total_cost
    last_prices[order.symbol] = float(price)
    _update_nav_order(order, entry_date, fill)
    _record_nav_fill_audit(
        fill_rows,
        order=order,
        trade_date=entry_date,
        trade_idx=trade_idx,
        capacity_notional=float(order.requested_notional),
        filled_notional=fill,
        transaction_cost=cost.total_cost,
        cost_breakdown=cost,
        valuation_time=entry_date,
    )
    return cost


def _ideal_sell_status(order: _NavOrder, price: float) -> str:
    if order.remaining_notional <= 1e-8:
        return "filled"
    return "missing_price" if not np.isfinite(price) else "partially_filled"


def _execute_ideal_buy_orders(
    *,
    buy_orders: list[_NavOrder],
    shares: dict[str, float],
    cash_ref: dict[str, float],
    entry_date: pd.Timestamp,
    trade_idx: int,
    tables: _ExecutionTables,
    config: ExecutionSimConfig,
    last_prices: dict[str, float],
    cost_rate: float,
    trade_fee_model: TradeFeeModel | None,
    order_rows: list[dict[str, Any]],
    fill_rows: list[dict[str, Any]],
) -> tuple[float, CostBreakdown]:
    valid_buy_orders: list[tuple[_NavOrder, float]] = []
    for order in buy_orders:
        price = _price_at(order.symbol, entry_date, tables.price_table)
        if np.isfinite(price):
            valid_buy_orders.append((order, price))
    total_cash_required = sum(
        float(order.remaining_notional)
        + _trade_fee(
            order.remaining_notional,
            side="buy",
            cost_rate=cost_rate,
            fee_model=trade_fee_model,
        ).total_cost
        for order, _ in valid_buy_orders
    )
    cash = max(float(cash_ref.get("cash", 0.0)), 0.0)
    if total_cash_required <= 0:
        scale = 0.0
    elif cash + 1e-6 >= total_cash_required:
        scale = 1.0
    else:
        scale = min(1.0, cash / total_cash_required)

    traded_notional = 0.0
    transaction_cost = CostBreakdown()
    for order in buy_orders:
        price = _price_at(order.symbol, entry_date, tables.price_table)
        fill = float(order.remaining_notional) * scale if np.isfinite(price) else 0.0
        if fill > 1e-8:
            cost = _apply_ideal_buy_fill(
                order=order,
                shares=shares,
                cash_ref=cash_ref,
                price=float(price),
                fill=fill,
                cost_rate=cost_rate,
                trade_fee_model=trade_fee_model,
                last_prices=last_prices,
                entry_date=entry_date,
                trade_idx=trade_idx,
                fill_rows=fill_rows,
            )
            traded_notional += fill
            transaction_cost = _add_breakdown(transaction_cost, cost)
        order.status = _ideal_buy_status(order, price)
        _append_nav_order_row(
            order_rows,
            order,
            trade_date=entry_date,
            participation_rate=config.participation_rate,
        )

    return float(traded_notional), transaction_cost


def _apply_ideal_buy_fill(
    *,
    order: _NavOrder,
    shares: dict[str, float],
    cash_ref: dict[str, float],
    price: float,
    fill: float,
    cost_rate: float,
    trade_fee_model: TradeFeeModel | None,
    last_prices: dict[str, float],
    entry_date: pd.Timestamp,
    trade_idx: int,
    fill_rows: list[dict[str, Any]],
) -> CostBreakdown:
    cost = _trade_fee(fill, side="buy", cost_rate=cost_rate, fee_model=trade_fee_model)
    shares[order.symbol] = float(shares.get(order.symbol, 0.0)) + fill / price
    cash_ref["cash"] = float(cash_ref.get("cash", 0.0)) - fill - cost.total_cost
    last_prices[order.symbol] = float(price)
    _update_nav_order(order, entry_date, fill)
    _record_nav_fill_audit(
        fill_rows,
        order=order,
        trade_date=entry_date,
        trade_idx=trade_idx,
        capacity_notional=float(order.requested_notional),
        filled_notional=fill,
        transaction_cost=cost.total_cost,
        cost_breakdown=cost,
        valuation_time=entry_date,
    )
    return cost


def _ideal_buy_status(order: _NavOrder, price: float) -> str:
    if order.remaining_notional <= 1e-8:
        return "filled"
    return "missing_price" if not np.isfinite(price) else "insufficient_cash"
