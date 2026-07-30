"""Shared dataclasses for execution simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..execution import DetailedTradeFeeModel

TradeFeeModel = DetailedTradeFeeModel

__all__ = [
    "_AdjustedNavLedger",
    "_AdjustedNavPlan",
    "_ExecutionTables",
    "_NavOrder",
    "_OrderSink",
    "_trade_fee",
    "describe_trade_fee_model",
]


def describe_trade_fee_model(
    fee_model: TradeFeeModel | None,
    *,
    portfolio_value: float | None = None,
) -> dict[str, Any]:
    if fee_model is None:
        return {"name": "bps"}
    effective_portfolio_value = (
        float(portfolio_value)
        if portfolio_value is not None and np.isfinite(portfolio_value) and portfolio_value > 0
        else float(fee_model.portfolio_value)
    )
    return {
        "name": "detailed",
        "buy_commission_bps": float(fee_model.buy_commission_bps),
        "sell_commission_bps": float(fee_model.sell_commission_bps),
        "sell_stamp_duty_bps": float(fee_model.sell_stamp_duty_bps),
        "transfer_fee_bps": float(fee_model.transfer_fee_bps),
        "min_commission": float(fee_model.min_commission),
        "buy_slippage_bps": float(fee_model.buy_slippage_bps),
        "sell_slippage_bps": float(fee_model.sell_slippage_bps),
        "portfolio_value": effective_portfolio_value,
    }


def _trade_fee(
    notional: float,
    *,
    side: str,
    cost_rate: float,
    fee_model: TradeFeeModel | None,
) -> float:
    if fee_model is None:
        return max(float(notional), 0.0) * max(float(cost_rate), 0.0)
    return fee_model.notional_cost(notional, side=side)


@dataclass(frozen=True)
class _ExecutionTables:
    trade_dates: list[pd.Timestamp]
    date_to_idx: dict[pd.Timestamp, int]
    price_table: pd.DataFrame
    buy_tradable_table: pd.DataFrame | None
    sell_tradable_table: pd.DataFrame | None
    liquidity_tables: dict[str, pd.DataFrame]


@dataclass(frozen=True)
class _OrderSink:
    order_rows: list[dict[str, Any]]
    fill_rows: list[dict[str, Any]]


@dataclass
class _NavOrder:
    rebalance_date: pd.Timestamp
    entry_date: pd.Timestamp
    side: str
    symbol: str
    requested_notional: float
    remaining_notional: float
    start_idx: int
    max_days: int
    zero_fill_days: int = 0
    filled_notional: float = 0.0
    first_fill_date: pd.Timestamp | None = None
    last_fill_date: pd.Timestamp | None = None
    fill_days: int = 0
    status: str | None = None
    requested_quantity: float | None = None
    remaining_quantity: float | None = None
    filled_quantity: float = 0.0


@dataclass(frozen=True)
class _AdjustedNavPlan:
    tables: _ExecutionTables
    targets_by_entry: dict[pd.Timestamp, tuple[pd.Timestamp, dict[str, float]]]
    next_entry_by_date: dict[pd.Timestamp, pd.Timestamp | None]
    start_idx: int
    cost_rate: float


@dataclass
class _AdjustedNavLedger:
    cash: float
    previous_nav: float
    target_cash_notional: float
    shares: dict[str, float]
    last_prices: dict[str, float]
    open_orders: list[_NavOrder]
    order_rows: list[dict[str, Any]]
    fill_rows: list[dict[str, Any]]
    daily_rows: list[dict[str, Any]]
