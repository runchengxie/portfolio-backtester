"""Price and liquidity capacity queries."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..execution import DetailedTradeFeeModel
from .config import (
    SELL_UNTIL_NEXT_REBALANCE,
    ExecutionSimConfig,
)

TradeFeeModel = DetailedTradeFeeModel

__all__ = [
    '_capacity_notional',
    '_capacity_weight',
    '_execution_window_dates',
    '_position_values_by_symbol',
    '_positions_value',
    '_price_at',
    '_refresh_last_prices',
    '_table_bool_at',
    '_table_float_at',
    '_valuation_price',
]

def _execution_window_dates(
    entry_date: pd.Timestamp,
    *,
    max_days: int | str,
    next_entry_date: pd.Timestamp | None,
    trade_dates: list[pd.Timestamp],
    date_to_idx: dict[pd.Timestamp, int],
) -> list[pd.Timestamp]:
    if entry_date not in date_to_idx:
        return []
    start_idx = date_to_idx[entry_date]
    if max_days == SELL_UNTIL_NEXT_REBALANCE:
        if next_entry_date is not None and next_entry_date in date_to_idx:
            end_idx = max(start_idx + 1, date_to_idx[next_entry_date])
        else:
            end_idx = len(trade_dates)
        return trade_dates[start_idx : min(end_idx, len(trade_dates))]

    end_idx = min(start_idx + int(max_days), len(trade_dates))
    return trade_dates[start_idx:end_idx]


def _capacity_weight(
    symbol: str,
    trade_date: pd.Timestamp,
    *,
    config: ExecutionSimConfig,
    price_table: pd.DataFrame,
    tradable_table: pd.DataFrame | None,
    liquidity_tables: dict[str, pd.DataFrame],
) -> float:
    price = _table_float_at(price_table, trade_date, symbol)
    if not np.isfinite(price) or price <= 0:
        return 0.0
    if tradable_table is not None and not _table_bool_at(tradable_table, trade_date, symbol):
        return 0.0

    liquidity_values: list[float] = []
    for column in config.liquidity_cols:
        table = liquidity_tables.get(column)
        if table is None:
            return 0.0
        value = _table_float_at(table, trade_date, symbol)
        if not np.isfinite(value) or value <= 0:
            return 0.0
        liquidity_values.append(float(value))
    if not liquidity_values:
        return 0.0
    liquidity = min(liquidity_values)
    notional = float(config.participation_rate) * liquidity
    return max(notional / float(config.portfolio_value), 0.0)


def _capacity_notional(
    symbol: str,
    trade_date: pd.Timestamp,
    *,
    config: ExecutionSimConfig,
    price_table: pd.DataFrame,
    tradable_table: pd.DataFrame | None,
    liquidity_tables: dict[str, pd.DataFrame],
) -> float:
    return _capacity_weight(
        symbol,
        trade_date,
        config=config,
        price_table=price_table,
        tradable_table=tradable_table,
        liquidity_tables=liquidity_tables,
    ) * float(config.portfolio_value)


def _price_at(
    symbol: str,
    trade_date: pd.Timestamp,
    price_table: pd.DataFrame,
) -> float:
    value = _table_float_at(price_table, trade_date, symbol)
    if not np.isfinite(value) or value <= 0:
        return np.nan
    return float(value)


def _table_float_at(table: pd.DataFrame, trade_date: pd.Timestamp, symbol: str) -> float:
    if table.empty or trade_date not in table.index or symbol not in table.columns:
        return np.nan
    try:
        value = table.at[trade_date, symbol]
    except (KeyError, ValueError):
        return np.nan
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan
    return result if np.isfinite(result) else np.nan


def _table_bool_at(table: pd.DataFrame, trade_date: pd.Timestamp, symbol: str) -> bool:
    if table.empty or trade_date not in table.index or symbol not in table.columns:
        return False
    try:
        return bool(table.at[trade_date, symbol])
    except (KeyError, ValueError):
        return False


def _valuation_price(
    symbol: str,
    trade_date: pd.Timestamp,
    price_table: pd.DataFrame,
    last_prices: dict[str, float],
) -> float:
    price = _price_at(symbol, trade_date, price_table)
    if np.isfinite(price):
        return float(price)
    return float(last_prices.get(symbol, np.nan))


def _refresh_last_prices(
    last_prices: dict[str, float],
    shares: dict[str, float],
    trade_date: pd.Timestamp,
    price_table: pd.DataFrame,
) -> None:
    for symbol in list(shares):
        price = _price_at(symbol, trade_date, price_table)
        if np.isfinite(price):
            last_prices[symbol] = float(price)


def _positions_value(
    shares: dict[str, float],
    trade_date: pd.Timestamp,
    price_table: pd.DataFrame,
    last_prices: dict[str, float],
) -> float:
    value = 0.0
    for symbol, quantity in shares.items():
        price = _valuation_price(symbol, trade_date, price_table, last_prices)
        if np.isfinite(price):
            value += float(quantity) * float(price)
    return float(value)


def _position_values_by_symbol(
    shares: dict[str, float],
    trade_date: pd.Timestamp,
    price_table: pd.DataFrame,
    last_prices: dict[str, float],
) -> dict[str, float]:
    values: dict[str, float] = {}
    for symbol, quantity in shares.items():
        price = _valuation_price(symbol, trade_date, price_table, last_prices)
        values[symbol] = float(quantity) * float(price) if np.isfinite(price) else 0.0
    return values


