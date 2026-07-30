"""Result aggregation and column schemas."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..execution import DetailedTradeFeeModel
from ..metrics import summarize_period_returns
from .config import (
    ExecutionSimConfig,
    describe_execution_sim_config,
)
from .models import (
    describe_trade_fee_model,
)
from .results import (
    ExecutionAdjustedNavResult,
    ExecutionSimResult,
)

TradeFeeModel = DetailedTradeFeeModel

__all__ = [
    "_daily_period_info",
    "_empty_adjusted_nav_result",
    "_empty_result",
    "_executed_daily_columns",
    "_fill_columns",
    "_format_date",
    "_nav_fill_columns",
    "_nav_order_columns",
    "_nav_side_fill_ratio",
    "_order_columns",
    "_side_fill_ratio",
    "_summarize_adjusted_nav",
    "_summarize_orders",
]


def _summarize_orders(
    config: ExecutionSimConfig,
    orders: pd.DataFrame,
    *,
    rebalances: int,
    final_cash_weight: float,
    final_invested_weight: float,
    status: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "enabled": bool(config.enabled),
        "status": status,
        "config": describe_execution_sim_config(config),
        "rebalances": int(rebalances),
        "orders": int(orders.shape[0]),
        "final_cash_weight": float(final_cash_weight),
        "final_invested_weight": float(final_invested_weight),
    }
    if orders.empty:
        summary.update(
            {
                "requested_notional": 0.0,
                "filled_notional": 0.0,
                "unfilled_notional": 0.0,
                "fill_ratio": np.nan,
                "buy_fill_ratio": np.nan,
                "sell_fill_ratio": np.nan,
                "unfilled_buy_notional": 0.0,
                "unfilled_sell_notional": 0.0,
                "abandoned_buy_orders": 0,
                "delayed_sell_orders": 0,
            }
        )
    else:
        requested = float(orders["requested_notional"].sum())
        filled = float(orders["filled_notional"].sum())
        unfilled = float(orders["unfilled_notional"].sum())
        buy_orders = orders[orders["side"] == "buy"]
        sell_orders = orders[orders["side"] == "sell"]
        summary.update(
            {
                "requested_notional": requested,
                "filled_notional": filled,
                "unfilled_notional": unfilled,
                "fill_ratio": filled / requested if requested > 0 else np.nan,
                "buy_fill_ratio": _side_fill_ratio(buy_orders),
                "sell_fill_ratio": _side_fill_ratio(sell_orders),
                "unfilled_buy_notional": float(buy_orders["unfilled_notional"].sum())
                if not buy_orders.empty
                else 0.0,
                "unfilled_sell_notional": float(sell_orders["unfilled_notional"].sum())
                if not sell_orders.empty
                else 0.0,
                "abandoned_buy_orders": int((buy_orders["status"] == "abandoned_zero_fill").sum())
                if not buy_orders.empty
                else 0,
                "delayed_sell_orders": int((sell_orders["status"] == "delayed_sell").sum())
                if not sell_orders.empty
                else 0,
            }
        )
    if extra:
        summary.update(extra)
    return summary


def _empty_result(
    config: ExecutionSimConfig,
    *,
    status: str,
    extra: dict[str, Any] | None = None,
) -> ExecutionSimResult:
    orders = pd.DataFrame(columns=_order_columns())
    fills = pd.DataFrame(columns=_fill_columns())
    summary = _summarize_orders(
        config,
        orders,
        rebalances=0,
        final_cash_weight=1.0,
        final_invested_weight=0.0,
        status=status,
        extra=extra,
    )
    return ExecutionSimResult(summary=summary, orders=orders, fills=fills)


def _empty_adjusted_nav_result(
    config: ExecutionSimConfig,
    *,
    status: str,
    extra: dict[str, Any] | None = None,
) -> ExecutionAdjustedNavResult:
    daily = pd.DataFrame(columns=_executed_daily_columns())
    orders = pd.DataFrame(columns=_nav_order_columns())
    fills = pd.DataFrame(columns=_nav_fill_columns())
    summary = {
        "enabled": bool(config.enabled),
        "status": status,
        "config": describe_execution_sim_config(config),
        "daily_rows": 0,
        "first_trade_date": None,
        "last_trade_date": None,
        "transaction_cost_bps": np.nan,
        "requested_notional": 0.0,
        "filled_notional": 0.0,
        "unfilled_notional": 0.0,
        "fill_ratio": np.nan,
        "buy_fill_ratio": np.nan,
        "sell_fill_ratio": np.nan,
        "avg_cash_weight": np.nan,
        "avg_target_cash_weight": np.nan,
        "avg_execution_shortfall_cash_weight": np.nan,
        "avg_gross_exposure": np.nan,
        "final_cash_weight": np.nan,
        "final_target_cash_weight": np.nan,
        "final_execution_shortfall_cash_weight": np.nan,
        "final_gross_exposure": np.nan,
        "stats": summarize_period_returns(pd.Series(dtype=float), [], 252),
    }
    if extra:
        summary.update(extra)
    return ExecutionAdjustedNavResult(summary=summary, daily=daily, orders=orders, fills=fills)


def _summarize_adjusted_nav(
    config: ExecutionSimConfig,
    *,
    daily: pd.DataFrame,
    orders: pd.DataFrame,
    transaction_cost_bps: float,
    trading_days_per_year: int,
    status: str,
    trade_fee_model: TradeFeeModel | None = None,
) -> dict[str, Any]:
    returns = (
        pd.Series(dtype=float)
        if daily.empty
        else pd.Series(
            pd.to_numeric(daily["executed_return"], errors="coerce").to_numpy(dtype=float),
            index=pd.to_datetime(daily["trade_date"], errors="coerce"),
            name="executed_return",
        ).dropna()
    )
    stats = summarize_period_returns(
        returns,
        _daily_period_info(len(returns)),
        int(trading_days_per_year),
    )
    requested = float(orders["requested_notional"].sum()) if not orders.empty else 0.0
    filled = float(orders["filled_notional"].sum()) if not orders.empty else 0.0
    unfilled = float(orders["unfilled_notional"].sum()) if not orders.empty else 0.0
    buy_orders = orders[orders["side"] == "buy"] if not orders.empty else pd.DataFrame()
    sell_orders = orders[orders["side"] == "sell"] if not orders.empty else pd.DataFrame()
    return {
        "enabled": bool(config.enabled),
        "status": status,
        "config": describe_execution_sim_config(config),
        "daily_rows": int(daily.shape[0]),
        "first_trade_date": None if daily.empty else str(daily["trade_date"].iloc[0]),
        "last_trade_date": None if daily.empty else str(daily["trade_date"].iloc[-1]),
        "transaction_cost_bps": float(transaction_cost_bps),
        "fee_model": describe_trade_fee_model(
            trade_fee_model,
            portfolio_value=config.portfolio_value,
        ),
        "requested_notional": requested,
        "filled_notional": filled,
        "unfilled_notional": unfilled,
        "fill_ratio": filled / requested if requested > 0 else np.nan,
        "buy_fill_ratio": _nav_side_fill_ratio(buy_orders),
        "sell_fill_ratio": _nav_side_fill_ratio(sell_orders),
        "avg_cash_weight": float(pd.to_numeric(daily["cash_weight"], errors="coerce").mean())
        if not daily.empty
        else np.nan,
        "avg_target_cash_weight": float(
            pd.to_numeric(daily["target_cash_weight"], errors="coerce").mean()
        )
        if not daily.empty and "target_cash_weight" in daily
        else np.nan,
        "avg_execution_shortfall_cash_weight": float(
            pd.to_numeric(daily["execution_shortfall_cash_weight"], errors="coerce").mean()
        )
        if not daily.empty and "execution_shortfall_cash_weight" in daily
        else np.nan,
        "avg_gross_exposure": float(pd.to_numeric(daily["gross_exposure"], errors="coerce").mean())
        if not daily.empty
        else np.nan,
        "final_cash_weight": float(daily["cash_weight"].iloc[-1]) if not daily.empty else np.nan,
        "final_target_cash_weight": float(daily["target_cash_weight"].iloc[-1])
        if not daily.empty and "target_cash_weight" in daily
        else np.nan,
        "final_execution_shortfall_cash_weight": float(
            daily["execution_shortfall_cash_weight"].iloc[-1]
        )
        if not daily.empty and "execution_shortfall_cash_weight" in daily
        else np.nan,
        "final_gross_exposure": float(daily["gross_exposure"].iloc[-1])
        if not daily.empty
        else np.nan,
        "stats": stats,
    }


def _daily_period_info(length: int) -> list[dict[str, int]]:
    return [{"entry_idx": idx, "exit_idx": idx + 1} for idx in range(max(int(length), 0))]


def _nav_side_fill_ratio(frame: pd.DataFrame) -> float:
    if frame.empty:
        return np.nan
    requested = float(frame["requested_notional"].sum())
    if requested <= 0:
        return np.nan
    return float(frame["filled_notional"].sum()) / requested


def _side_fill_ratio(frame: pd.DataFrame) -> float:
    if frame.empty:
        return np.nan
    requested = float(frame["requested_notional"].sum())
    if requested <= 0:
        return np.nan
    return float(frame["filled_notional"].sum()) / requested


def _format_date(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.to_datetime(value).strftime("%Y%m%d")


def _order_columns() -> list[str]:
    return [
        "rebalance_date",
        "entry_date",
        "side",
        "symbol",
        "requested_weight",
        "filled_weight",
        "unfilled_weight",
        "requested_notional",
        "filled_notional",
        "unfilled_notional",
        "fill_ratio",
        "status",
        "first_fill_date",
        "last_fill_date",
        "fill_days",
        "max_days",
        "zero_fill_days",
        "participation_rate",
    ]


def _fill_columns() -> list[str]:
    return [
        "rebalance_date",
        "entry_date",
        "trade_date",
        "day_number",
        "side",
        "symbol",
        "remaining_before_weight",
        "capacity_weight",
        "filled_weight",
        "capacity_notional",
        "filled_notional",
    ]


def _executed_daily_columns() -> list[str]:
    return [
        "trade_date",
        "executed_return",
        "executed_nav",
        "portfolio_value",
        "cash",
        "invested_value",
        "cash_weight",
        "target_cash_weight",
        "execution_shortfall_cash_weight",
        "gross_exposure",
        "traded_notional",
        "transaction_cost",
        "open_orders",
    ]


def _nav_order_columns() -> list[str]:
    return [
        "rebalance_date",
        "entry_date",
        "side",
        "symbol",
        "requested_notional",
        "filled_notional",
        "unfilled_notional",
        "fill_ratio",
        "status",
        "first_fill_date",
        "last_fill_date",
        "closed_date",
        "fill_days",
        "max_days",
        "zero_fill_days",
        "participation_rate",
    ]


def _nav_fill_columns() -> list[str]:
    return [
        "rebalance_date",
        "entry_date",
        "trade_date",
        "day_number",
        "side",
        "symbol",
        "remaining_before_notional",
        "capacity_notional",
        "filled_notional",
        "transaction_cost",
    ]
