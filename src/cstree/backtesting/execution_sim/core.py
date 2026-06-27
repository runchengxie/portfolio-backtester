"""Order-level capacity execution simulation for rebalance targets."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..execution import DetailedTradeFeeModel
from ..metrics import summarize_period_returns
from .config import (
    SELL_UNTIL_NEXT_REBALANCE,
    ExecutionSimConfig,
    describe_execution_sim_config,
    required_execution_sim_columns,
)
from .results import ExecutionAdjustedNavResult, ExecutionSimResult

TradeFeeModel = DetailedTradeFeeModel


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


def simulate_capacity_execution(
    positions: pd.DataFrame | None,
    pricing_data: pd.DataFrame | None,
    config: ExecutionSimConfig,
    *,
    price_col: str,
    tradable_col: str | None = None,
    buy_tradable_col: str | None = None,
    sell_tradable_col: str | None = None,
) -> ExecutionSimResult:
    if not config.enabled:
        return _empty_result(config, status="disabled")
    if positions is None or positions.empty:
        return _empty_result(config, status="no_positions")
    if pricing_data is None or pricing_data.empty:
        return _empty_result(config, status="no_pricing_data")

    work_positions, status, extra = _prepare_long_only_execution_positions(positions)
    if status is not None or work_positions is None:
        return _empty_result(config, status=status or "no_usable_positions", extra=extra)

    execution_tables, status, extra = _prepare_execution_tables(
        pricing_data,
        config,
        price_col=price_col,
        tradable_col=tradable_col,
        buy_tradable_col=buy_tradable_col,
        sell_tradable_col=sell_tradable_col,
    )
    if status is not None or execution_tables is None:
        return _empty_result(config, status=status or "no_trade_dates", extra=extra)

    orders, fills, cash_weight, current_weights, rebalance_count = _run_capacity_rebalances(
        work_positions,
        tables=execution_tables,
        config=config,
    )
    summary = _summarize_orders(
        config,
        orders,
        rebalances=rebalance_count,
        final_cash_weight=cash_weight,
        final_invested_weight=sum(current_weights.values()),
        status="ok",
    )
    return ExecutionSimResult(summary=summary, orders=orders, fills=fills)


def _run_capacity_rebalances(
    work_positions: pd.DataFrame,
    *,
    tables: _ExecutionTables,
    config: ExecutionSimConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, float, dict[str, float], int]:
    targets_by_rebalance = _build_targets_by_rebalance(work_positions)
    current_weights: dict[str, float] = {}
    cash_weight = 1.0
    order_rows: list[dict[str, Any]] = []
    fill_rows: list[dict[str, Any]] = []
    order_sink = _OrderSink(order_rows=order_rows, fill_rows=fill_rows)

    for idx, target in enumerate(targets_by_rebalance):
        cash_weight = _execute_capacity_rebalance(
            target,
            target_idx=idx,
            targets_by_rebalance=targets_by_rebalance,
            current_weights=current_weights,
            cash_weight=cash_weight,
            config=config,
            tables=tables,
            sink=order_sink,
        )

    orders = pd.DataFrame(order_rows, columns=_order_columns())
    fills = pd.DataFrame(fill_rows, columns=_fill_columns())
    return orders, fills, cash_weight, current_weights, len(targets_by_rebalance)


def _execute_capacity_rebalance(
    target: tuple[pd.Timestamp, dict[str, Any]],
    *,
    target_idx: int,
    targets_by_rebalance: list[tuple[pd.Timestamp, dict[str, Any]]],
    current_weights: dict[str, float],
    cash_weight: float,
    config: ExecutionSimConfig,
    tables: _ExecutionTables,
    sink: _OrderSink,
) -> float:
    rebalance_date, target_info = target
    entry_date = target_info["entry_date"]
    if entry_date not in tables.date_to_idx:
        return cash_weight

    next_entry_date = (
        targets_by_rebalance[target_idx + 1][1]["entry_date"]
        if target_idx + 1 < len(targets_by_rebalance)
        else None
    )
    target_weights = target_info["weights"]
    symbols = sorted(set(current_weights) | set(target_weights))
    deltas = {
        symbol: float(target_weights.get(symbol, 0.0) - current_weights.get(symbol, 0.0))
        for symbol in symbols
    }
    sell_requests = {symbol: -delta for symbol, delta in deltas.items() if delta < -1e-12}
    if sell_requests:
        cash_weight = _execute_sell_orders(
            rebalance_date=rebalance_date,
            entry_date=entry_date,
            next_entry_date=next_entry_date,
            requests=sell_requests,
            current_weights=current_weights,
            cash_weight=cash_weight,
            config=config,
            tables=tables,
            sink=sink,
        )

    buy_requests = {
        symbol: max(float(target_weights.get(symbol, 0.0) - current_weights.get(symbol, 0.0)), 0.0)
        for symbol in target_weights
    }
    buy_requests = {symbol: amount for symbol, amount in buy_requests.items() if amount > 1e-12}
    if buy_requests:
        return _execute_buy_orders(
            rebalance_date=rebalance_date,
            entry_date=entry_date,
            requests=buy_requests,
            current_weights=current_weights,
            cash_weight=cash_weight,
            config=config,
            tables=tables,
            sink=sink,
        )
    return cash_weight


def _prepare_long_only_execution_positions(
    positions: pd.DataFrame,
) -> tuple[pd.DataFrame | None, str | None, dict[str, Any] | None]:
    work_positions = positions.copy()
    if "side" in work_positions.columns:
        unsupported_side = work_positions["side"].astype(str).str.lower().eq("short").any()
        if unsupported_side:
            return None, "skipped_long_short_not_supported", None
    work_positions["weight"] = pd.to_numeric(work_positions["weight"], errors="coerce")
    if (work_positions["weight"] < 0).any():
        return None, "skipped_negative_weights_not_supported", None
    work_positions["rebalance_date"] = pd.to_datetime(
        work_positions["rebalance_date"], errors="coerce"
    )
    work_positions["entry_date"] = pd.to_datetime(work_positions["entry_date"], errors="coerce")
    work_positions = work_positions.dropna(subset=["rebalance_date", "entry_date", "symbol"])
    work_positions = work_positions[work_positions["weight"].notna()].copy()
    if work_positions.empty:
        return None, "no_usable_positions", None
    return work_positions, None, None


def _prepare_execution_tables(
    pricing_data: pd.DataFrame,
    config: ExecutionSimConfig,
    *,
    price_col: str,
    tradable_col: str | None,
    buy_tradable_col: str | None,
    sell_tradable_col: str | None,
) -> tuple[_ExecutionTables | None, str | None, dict[str, Any] | None]:
    pricing = pricing_data.drop_duplicates(subset=["trade_date", "symbol"]).copy()
    pricing["trade_date"] = pd.to_datetime(pricing["trade_date"], errors="coerce")
    pricing = pricing.dropna(subset=["trade_date", "symbol"])
    required_cols = required_execution_sim_columns(
        config,
        price_col=price_col,
        tradable_col=tradable_col if tradable_col in pricing.columns else None,
    )
    missing_cols = sorted(col for col in required_cols if col not in pricing.columns)
    if missing_cols:
        return None, "missing_pricing_columns", {"missing_pricing_columns": missing_cols}

    tables = _build_execution_tables(
        pricing,
        config,
        price_col=price_col,
        tradable_col=tradable_col,
        buy_tradable_col=buy_tradable_col,
        sell_tradable_col=sell_tradable_col,
    )
    if not tables.trade_dates:
        return None, "no_trade_dates", None
    return tables, None, None


def _build_adjusted_nav_plan(
    work_positions: pd.DataFrame,
    *,
    tables: _ExecutionTables,
    cost_rate: float,
) -> tuple[_AdjustedNavPlan | None, str | None]:
    targets_by_rebalance = _build_targets_by_rebalance(work_positions)
    targets_by_entry = {
        info["entry_date"]: (rebalance_date, info["weights"])
        for rebalance_date, info in targets_by_rebalance
        if info["entry_date"] in tables.date_to_idx
    }
    if not targets_by_entry:
        return None, "no_executable_entry_dates"
    entry_dates = sorted(targets_by_entry)
    next_entry_by_date = {
        entry_date: entry_dates[idx + 1] if idx + 1 < len(entry_dates) else None
        for idx, entry_date in enumerate(entry_dates)
    }
    return (
        _AdjustedNavPlan(
            tables=tables,
            targets_by_entry=targets_by_entry,
            next_entry_by_date=next_entry_by_date,
            start_idx=tables.date_to_idx[entry_dates[0]],
            cost_rate=cost_rate,
        ),
        None,
    )


def _initial_adjusted_nav_ledger(config: ExecutionSimConfig) -> _AdjustedNavLedger:
    initial_value = float(config.portfolio_value)
    return _AdjustedNavLedger(
        cash=initial_value,
        previous_nav=initial_value,
        target_cash_notional=0.0,
        shares={},
        last_prices={},
        open_orders=[],
        order_rows=[],
        fill_rows=[],
        daily_rows=[],
    )


def _start_adjusted_nav_target_orders(
    ledger: _AdjustedNavLedger,
    *,
    plan: _AdjustedNavPlan,
    trade_date: pd.Timestamp,
    trade_idx: int,
    nav_before_orders: float,
    config: ExecutionSimConfig,
) -> None:
    _finalize_open_nav_orders(
        ledger.open_orders,
        ledger.order_rows,
        trade_date=trade_date,
        participation_rate=config.participation_rate,
        status_by_side={"buy": "cancelled_new_target", "sell": "replaced_new_target"},
    )
    ledger.open_orders = []
    rebalance_date, target_weights = plan.targets_by_entry[trade_date]
    ledger.target_cash_notional = _target_cash_notional(target_weights, nav_before_orders)
    ledger.open_orders = _build_nav_orders_for_target(
        rebalance_date=rebalance_date,
        entry_date=trade_date,
        next_entry_date=plan.next_entry_by_date[trade_date],
        target_weights=target_weights,
        shares=ledger.shares,
        cash=ledger.cash,
        nav=nav_before_orders,
        trade_idx=trade_idx,
        tables=plan.tables,
        config=config,
        last_prices=ledger.last_prices,
    )


def _retain_open_adjusted_nav_orders(
    ledger: _AdjustedNavLedger,
    *,
    trade_date: pd.Timestamp,
    trade_idx: int,
    config: ExecutionSimConfig,
) -> None:
    still_open: list[_NavOrder] = []
    for order in ledger.open_orders:
        day_number = trade_idx - order.start_idx + 1
        if order.remaining_notional <= 1e-8:
            order.status = "filled"
            _append_nav_order_row(
                ledger.order_rows,
                order,
                trade_date=trade_date,
                participation_rate=config.participation_rate,
            )
        elif order.side == "buy" and _nav_order_should_abort_buy(order, config):
            order.status = "abandoned_zero_fill"
            _append_nav_order_row(
                ledger.order_rows,
                order,
                trade_date=trade_date,
                participation_rate=config.participation_rate,
            )
        elif day_number >= order.max_days:
            order.status = "cancelled_buy_deadline" if order.side == "buy" else "delayed_sell"
            _append_nav_order_row(
                ledger.order_rows,
                order,
                trade_date=trade_date,
                participation_rate=config.participation_rate,
            )
        else:
            still_open.append(order)
    ledger.open_orders = still_open


def _append_adjusted_nav_daily_row(
    ledger: _AdjustedNavLedger,
    *,
    plan: _AdjustedNavPlan,
    trade_date: pd.Timestamp,
    traded_notional: float,
    transaction_cost: float,
    config: ExecutionSimConfig,
) -> None:
    current_value = _positions_value(
        ledger.shares,
        trade_date,
        plan.tables.price_table,
        ledger.last_prices,
    )
    nav_after_orders = ledger.cash + current_value
    daily_return = (
        nav_after_orders / ledger.previous_nav - 1.0 if ledger.previous_nav > 0 else np.nan
    )
    ledger.previous_nav = nav_after_orders
    cash_weight, target_cash_weight, shortfall_cash_weight = _cash_weight_breakdown(
        cash=ledger.cash,
        target_cash_notional=ledger.target_cash_notional,
        nav=nav_after_orders,
    )
    ledger.daily_rows.append(
        {
            "trade_date": _format_date(trade_date),
            "executed_return": float(daily_return),
            "executed_nav": float(nav_after_orders / float(config.portfolio_value)),
            "portfolio_value": float(nav_after_orders),
            "cash": float(ledger.cash),
            "invested_value": float(current_value),
            "cash_weight": cash_weight,
            "target_cash_weight": target_cash_weight,
            "execution_shortfall_cash_weight": shortfall_cash_weight,
            "gross_exposure": float(current_value / nav_after_orders)
            if nav_after_orders > 0
            else np.nan,
            "traded_notional": float(traded_notional),
            "transaction_cost": float(transaction_cost),
            "open_orders": len(ledger.open_orders),
        }
    )


def _process_adjusted_nav_trade_day(
    ledger: _AdjustedNavLedger,
    *,
    plan: _AdjustedNavPlan,
    trade_idx: int,
    config: ExecutionSimConfig,
    trade_fee_model: TradeFeeModel | None,
) -> None:
    trade_date = plan.tables.trade_dates[trade_idx]
    _refresh_last_prices(ledger.last_prices, ledger.shares, trade_date, plan.tables.price_table)
    nav_before_orders = ledger.cash + _positions_value(
        ledger.shares,
        trade_date,
        plan.tables.price_table,
        ledger.last_prices,
    )
    if trade_date in plan.targets_by_entry:
        _start_adjusted_nav_target_orders(
            ledger,
            plan=plan,
            trade_date=trade_date,
            trade_idx=trade_idx,
            nav_before_orders=nav_before_orders,
            config=config,
        )

    cash_box = {"cash": ledger.cash}
    traded_notional, transaction_cost = _execute_nav_orders_for_day(
        open_orders=ledger.open_orders,
        shares=ledger.shares,
        cash_ref=cash_box,
        trade_date=trade_date,
        trade_idx=trade_idx,
        tables=plan.tables,
        config=config,
        cost_rate=plan.cost_rate,
        trade_fee_model=trade_fee_model,
        fill_rows=ledger.fill_rows,
    )
    ledger.cash = float(cash_box["cash"])
    _retain_open_adjusted_nav_orders(
        ledger,
        trade_date=trade_date,
        trade_idx=trade_idx,
        config=config,
    )
    _append_adjusted_nav_daily_row(
        ledger,
        plan=plan,
        trade_date=trade_date,
        traded_notional=traded_notional,
        transaction_cost=transaction_cost,
        config=config,
    )


def _run_adjusted_nav_ledger(
    *,
    plan: _AdjustedNavPlan,
    config: ExecutionSimConfig,
    trade_fee_model: TradeFeeModel | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ledger = _initial_adjusted_nav_ledger(config)
    for trade_idx in range(plan.start_idx, len(plan.tables.trade_dates)):
        _process_adjusted_nav_trade_day(
            ledger,
            plan=plan,
            trade_idx=trade_idx,
            config=config,
            trade_fee_model=trade_fee_model,
        )

    if ledger.open_orders:
        final_date = plan.tables.trade_dates[-1]
        _finalize_open_nav_orders(
            ledger.open_orders,
            ledger.order_rows,
            trade_date=final_date,
            participation_rate=config.participation_rate,
            status_by_side={"buy": "cancelled_buy_deadline", "sell": "delayed_sell"},
        )

    daily = pd.DataFrame(ledger.daily_rows, columns=_executed_daily_columns())
    orders = pd.DataFrame(ledger.order_rows, columns=_nav_order_columns())
    fills = pd.DataFrame(ledger.fill_rows, columns=_nav_fill_columns())
    return daily, orders, fills


def simulate_execution_adjusted_nav(
    positions: pd.DataFrame | None,
    pricing_data: pd.DataFrame | None,
    config: ExecutionSimConfig,
    *,
    price_col: str,
    tradable_col: str | None = None,
    buy_tradable_col: str | None = None,
    sell_tradable_col: str | None = None,
    transaction_cost_bps: float = 0.0,
    trading_days_per_year: int = 252,
    trade_fee_model: TradeFeeModel | None = None,
) -> ExecutionAdjustedNavResult:
    if not config.enabled:
        return _empty_adjusted_nav_result(config, status="disabled")
    if positions is None or positions.empty:
        return _empty_adjusted_nav_result(config, status="no_positions")
    if pricing_data is None or pricing_data.empty:
        return _empty_adjusted_nav_result(config, status="no_pricing_data")

    work_positions, status, extra = _prepare_long_only_execution_positions(positions)
    if status is not None:
        return _empty_adjusted_nav_result(config, status=status, extra=extra)

    tables, status, extra = _prepare_execution_tables(
        pricing_data,
        config,
        price_col=price_col,
        tradable_col=tradable_col,
        buy_tradable_col=buy_tradable_col,
        sell_tradable_col=sell_tradable_col,
    )
    if status is not None or tables is None:
        return _empty_adjusted_nav_result(config, status=status or "no_trade_dates", extra=extra)

    plan, status = _build_adjusted_nav_plan(
        work_positions,
        tables=tables,
        cost_rate=max(float(transaction_cost_bps), 0.0) / 10_000.0,
    )
    if status is not None or plan is None:
        return _empty_adjusted_nav_result(config, status=status or "no_executable_entry_dates")

    daily, orders, fills = _run_adjusted_nav_ledger(
        plan=plan,
        config=config,
        trade_fee_model=trade_fee_model,
    )
    summary = _summarize_adjusted_nav(
        config,
        daily=daily,
        orders=orders,
        transaction_cost_bps=transaction_cost_bps,
        trading_days_per_year=trading_days_per_year,
        status="ok",
        trade_fee_model=trade_fee_model,
    )
    return ExecutionAdjustedNavResult(summary=summary, daily=daily, orders=orders, fills=fills)


def simulate_ideal_daily_nav(
    positions: pd.DataFrame | None,
    pricing_data: pd.DataFrame | None,
    *,
    price_col: str,
    transaction_cost_bps: float = 0.0,
    trading_days_per_year: int = 252,
    portfolio_value: float = 1_000_000.0,
    trade_fee_model: TradeFeeModel | None = None,
) -> ExecutionAdjustedNavResult:
    """Daily NAV for immediate, fully liquid rebalances to target weights."""
    config = ExecutionSimConfig(
        enabled=True,
        portfolio_value=float(portfolio_value),
        participation_rate=1.0,
        liquidity_cols=(),
        buy_max_days=1,
        sell_max_days=1,
        zero_fill_abort_days_buy=None,
    )
    if positions is None or positions.empty:
        return _empty_adjusted_nav_result(config, status="no_positions")
    if pricing_data is None or pricing_data.empty:
        return _empty_adjusted_nav_result(config, status="no_pricing_data")

    work_positions, status, extra = _prepare_long_only_execution_positions(positions)
    if status is not None:
        return _empty_adjusted_nav_result(config, status=status, extra=extra)
    tables, targets_by_entry, status, extra = _prepare_ideal_nav_targets(
        work_positions,
        pricing_data,
        config=config,
        price_col=price_col,
    )
    if status is not None:
        return _empty_adjusted_nav_result(config, status=status, extra=extra)

    daily, orders, fills = _run_ideal_daily_nav_ledger(
        config=config,
        tables=tables,
        targets_by_entry=targets_by_entry,
        cost_rate=max(float(transaction_cost_bps), 0.0) / 10_000.0,
        trade_fee_model=trade_fee_model,
    )
    summary = _summarize_adjusted_nav(
        config,
        daily=daily,
        orders=orders,
        transaction_cost_bps=transaction_cost_bps,
        trading_days_per_year=trading_days_per_year,
        status="ok",
        trade_fee_model=trade_fee_model,
    )
    summary["mode"] = "ideal_daily_nav"
    return ExecutionAdjustedNavResult(summary=summary, daily=daily, orders=orders, fills=fills)


def _prepare_ideal_nav_targets(
    work_positions: pd.DataFrame | None,
    pricing_data: pd.DataFrame,
    *,
    config: ExecutionSimConfig,
    price_col: str,
) -> tuple[
    _ExecutionTables | None,
    dict[pd.Timestamp, tuple[pd.Timestamp, dict[str, float]]],
    str | None,
    dict[str, Any] | None,
]:
    if work_positions is None:
        return None, {}, "no_usable_positions", None
    required_columns = {"trade_date", "symbol", price_col}
    missing_columns = sorted(col for col in required_columns if col not in pricing_data.columns)
    if missing_columns:
        return None, {}, "missing_pricing_columns", {"missing_pricing_columns": missing_columns}

    pricing = pricing_data.drop_duplicates(subset=["trade_date", "symbol"]).copy()
    pricing["trade_date"] = pd.to_datetime(pricing["trade_date"], errors="coerce")
    pricing = pricing.dropna(subset=["trade_date", "symbol"])
    tables = _build_execution_tables(
        pricing,
        config,
        price_col=price_col,
        tradable_col=None,
        buy_tradable_col=None,
        sell_tradable_col=None,
    )
    if not tables.trade_dates:
        return None, {}, "no_trade_dates", None

    targets_by_rebalance = _build_targets_by_rebalance(work_positions)
    targets_by_entry = {
        info["entry_date"]: (rebalance_date, info["weights"])
        for rebalance_date, info in targets_by_rebalance
        if info["entry_date"] in tables.date_to_idx
    }
    if not targets_by_entry:
        return None, {}, "no_executable_entry_dates", None
    return tables, targets_by_entry, None, None


def _run_ideal_daily_nav_ledger(
    *,
    config: ExecutionSimConfig,
    tables: _ExecutionTables | None,
    targets_by_entry: dict[pd.Timestamp, tuple[pd.Timestamp, dict[str, float]]],
    cost_rate: float,
    trade_fee_model: TradeFeeModel | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if tables is None:
        return (
            pd.DataFrame(columns=_executed_daily_columns()),
            pd.DataFrame(columns=_nav_order_columns()),
            pd.DataFrame(columns=_nav_fill_columns()),
        )
    first_entry = sorted(targets_by_entry)[0]
    start_idx = tables.date_to_idx[first_entry]

    cash = float(config.portfolio_value)
    shares: dict[str, float] = {}
    last_prices: dict[str, float] = {}
    order_rows: list[dict[str, Any]] = []
    fill_rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    previous_nav = float(config.portfolio_value)

    for trade_idx in range(start_idx, len(tables.trade_dates)):
        trade_date = tables.trade_dates[trade_idx]
        _refresh_last_prices(last_prices, shares, trade_date, tables.price_table)
        nav_before_orders = cash + _positions_value(
            shares,
            trade_date,
            tables.price_table,
            last_prices,
        )
        traded_notional = 0.0
        transaction_cost = 0.0

        if trade_date in targets_by_entry:
            rebalance_date, target_weights = targets_by_entry[trade_date]
            cash_ref = {"cash": cash}
            traded_notional, transaction_cost = _rebalance_ideal_target(
                rebalance_date=rebalance_date,
                entry_date=trade_date,
                target_weights=target_weights,
                shares=shares,
                cash_ref=cash_ref,
                nav=nav_before_orders,
                trade_idx=trade_idx,
                tables=tables,
                config=config,
                last_prices=last_prices,
                cost_rate=cost_rate,
                trade_fee_model=trade_fee_model,
                order_rows=order_rows,
                fill_rows=fill_rows,
            )
            cash = float(cash_ref["cash"])

        current_value = _positions_value(shares, trade_date, tables.price_table, last_prices)
        nav_after_orders = cash + current_value
        daily_return = nav_after_orders / previous_nav - 1.0 if previous_nav > 0 else np.nan
        previous_nav = nav_after_orders
        daily_rows.append(
            _ideal_daily_nav_row(
                trade_date=trade_date,
                daily_return=daily_return,
                nav_after_orders=nav_after_orders,
                current_value=current_value,
                cash=cash,
                traded_notional=traded_notional,
                transaction_cost=transaction_cost,
                portfolio_value=config.portfolio_value,
            )
        )

    daily = pd.DataFrame(daily_rows, columns=_executed_daily_columns())
    orders = pd.DataFrame(order_rows, columns=_nav_order_columns())
    fills = pd.DataFrame(fill_rows, columns=_nav_fill_columns())
    return daily, orders, fills


def _ideal_daily_nav_row(
    *,
    trade_date: pd.Timestamp,
    daily_return: float,
    nav_after_orders: float,
    current_value: float,
    cash: float,
    traded_notional: float,
    transaction_cost: float,
    portfolio_value: float,
) -> dict[str, Any]:
    cash_weight = float(cash / nav_after_orders) if nav_after_orders > 0 else np.nan
    return {
        "trade_date": _format_date(trade_date),
        "executed_return": float(daily_return),
        "executed_nav": float(nav_after_orders / float(portfolio_value)),
        "portfolio_value": float(nav_after_orders),
        "cash": float(cash),
        "invested_value": float(current_value),
        "cash_weight": cash_weight,
        "target_cash_weight": cash_weight,
        "execution_shortfall_cash_weight": 0.0 if np.isfinite(cash_weight) else np.nan,
        "gross_exposure": float(current_value / nav_after_orders)
        if nav_after_orders > 0
        else np.nan,
        "traded_notional": float(traded_notional),
        "transaction_cost": float(transaction_cost),
        "open_orders": 0,
    }


def _build_execution_tables(
    pricing: pd.DataFrame,
    config: ExecutionSimConfig,
    *,
    price_col: str,
    tradable_col: str | None,
    buy_tradable_col: str | None,
    sell_tradable_col: str | None,
) -> _ExecutionTables:
    trade_dates = sorted(pd.to_datetime(pricing["trade_date"].unique()))
    date_to_idx = {date: idx for idx, date in enumerate(trade_dates)}
    price_table = pricing.pivot(index="trade_date", columns="symbol", values=price_col)
    tradable_table = _build_tradable_table(pricing, tradable_col)
    buy_tradable_table = _build_tradable_table(pricing, buy_tradable_col)
    sell_tradable_table = _build_tradable_table(pricing, sell_tradable_col)
    if buy_tradable_table is None:
        buy_tradable_table = tradable_table
    if sell_tradable_table is None:
        sell_tradable_table = tradable_table
    liquidity_tables = {
        col: pricing.pivot(index="trade_date", columns="symbol", values=col)
        for col in config.liquidity_cols
    }
    return _ExecutionTables(
        trade_dates=trade_dates,
        date_to_idx=date_to_idx,
        price_table=price_table,
        buy_tradable_table=buy_tradable_table,
        sell_tradable_table=sell_tradable_table,
        liquidity_tables=liquidity_tables,
    )


def _build_tradable_table(
    pricing: pd.DataFrame,
    tradable_col: str | None,
) -> pd.DataFrame | None:
    if not tradable_col or tradable_col not in pricing.columns:
        return None
    table = pricing.pivot(index="trade_date", columns="symbol", values=tradable_col)
    return table.mask(table.isna(), False).astype(bool)


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
    if tradable_table is not None:
        if not _table_bool_at(tradable_table, trade_date, symbol):
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


def _target_cash_notional(target_weights: Mapping[str, float], nav: float) -> float:
    if not np.isfinite(float(nav)) or float(nav) <= 0:
        return 0.0
    target_gross = sum(
        max(float(weight), 0.0) for weight in target_weights.values() if np.isfinite(float(weight))
    )
    return max(1.0 - float(target_gross), 0.0) * float(nav)


def _cash_weight_breakdown(
    *,
    cash: float,
    target_cash_notional: float,
    nav: float,
) -> tuple[float, float, float]:
    if not np.isfinite(float(nav)) or float(nav) <= 0:
        return np.nan, np.nan, np.nan
    cash_weight = max(float(cash), 0.0) / float(nav)
    target_cash_weight = min(
        max(float(target_cash_notional), 0.0) / float(nav),
        1.0,
    )
    return (
        float(cash_weight),
        float(target_cash_weight),
        float(max(cash_weight - target_cash_weight, 0.0)),
    )


def _cost_adjusted_target_notional(
    *,
    current_values: Mapping[str, float],
    target_weights: Mapping[str, float],
    nav: float,
    cost_rate: float,
) -> dict[str, float]:
    clean_weights = {
        str(symbol): max(float(weight), 0.0)
        for symbol, weight in target_weights.items()
        if pd.notna(symbol) and np.isfinite(float(weight)) and float(weight) > 0
    }
    if not clean_weights or nav <= 0:
        return {}
    if cost_rate <= 0:
        return {symbol: weight * float(nav) for symbol, weight in clean_weights.items()}

    clean_current = {
        str(symbol): max(float(value), 0.0)
        for symbol, value in current_values.items()
        if pd.notna(symbol) and np.isfinite(float(value)) and float(value) > 0
    }
    symbols = set(clean_current) | set(clean_weights)

    def required_cost(final_nav: float) -> float:
        turnover = 0.0
        for symbol in symbols:
            current_notional = clean_current.get(symbol, 0.0)
            target_notional = clean_weights.get(symbol, 0.0) * final_nav
            turnover += abs(target_notional - current_notional)
        return turnover * float(cost_rate)

    lower = 0.0
    upper = float(nav)
    for _ in range(64):
        mid = (lower + upper) / 2.0
        if mid + required_cost(mid) <= nav:
            lower = mid
        else:
            upper = mid
    return {symbol: weight * lower for symbol, weight in clean_weights.items()}


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
) -> tuple[float, float]:
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
    return float(sell_traded + buy_traded), float(sell_cost + buy_cost)


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
) -> tuple[float, float]:
    traded_notional = 0.0
    transaction_cost = 0.0
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
            transaction_cost += cost
        order.status = _ideal_sell_status(order, price)
        _append_nav_order_row(
            order_rows,
            order,
            trade_date=entry_date,
            participation_rate=config.participation_rate,
        )
    return float(traded_notional), float(transaction_cost)


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
) -> float:
    held_quantity = max(float(shares.get(order.symbol, 0.0)), 0.0)
    shares[order.symbol] = max(held_quantity - fill / price, 0.0)
    if shares[order.symbol] <= 1e-10:
        shares.pop(order.symbol, None)
    cost = _trade_fee(fill, side="sell", cost_rate=cost_rate, fee_model=trade_fee_model)
    cash_ref["cash"] = float(cash_ref.get("cash", 0.0)) + fill - cost
    last_prices[order.symbol] = float(price)
    _update_nav_order(order, entry_date, fill)
    _record_nav_fill(
        fill_rows,
        order=order,
        trade_date=entry_date,
        trade_idx=trade_idx,
        capacity_notional=float(order.requested_notional),
        filled_notional=fill,
        transaction_cost=cost,
    )
    return float(cost)


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
) -> tuple[float, float]:
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
        )
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
    transaction_cost = 0.0
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
            transaction_cost += cost
        order.status = _ideal_buy_status(order, price)
        _append_nav_order_row(
            order_rows,
            order,
            trade_date=entry_date,
            participation_rate=config.participation_rate,
        )

    return float(traded_notional), float(transaction_cost)


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
) -> float:
    cost = _trade_fee(fill, side="buy", cost_rate=cost_rate, fee_model=trade_fee_model)
    shares[order.symbol] = float(shares.get(order.symbol, 0.0)) + fill / price
    cash_ref["cash"] = float(cash_ref.get("cash", 0.0)) - fill - cost
    last_prices[order.symbol] = float(price)
    _update_nav_order(order, entry_date, fill)
    _record_nav_fill(
        fill_rows,
        order=order,
        trade_date=entry_date,
        trade_idx=trade_idx,
        capacity_notional=float(order.requested_notional),
        filled_notional=fill,
        transaction_cost=cost,
    )
    return float(cost)


def _ideal_buy_status(order: _NavOrder, price: float) -> str:
    if order.remaining_notional <= 1e-8:
        return "filled"
    return "missing_price" if not np.isfinite(price) else "insufficient_cash"


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
        [item for item in open_orders if item.side == "sell" and item.remaining_notional > 1e-8],
        key=lambda item: item.symbol,
    ):
        price = _price_at(order.symbol, trade_date, tables.price_table)
        if not np.isfinite(price):
            continue
        held_quantity = max(float(shares.get(order.symbol, 0.0)), 0.0)
        held_notional = held_quantity * float(price)
        capacity = _capacity_notional(
            order.symbol,
            trade_date,
            config=config,
            price_table=tables.price_table,
            tradable_table=tables.sell_tradable_table,
            liquidity_tables=tables.liquidity_tables,
        )
        fill = min(float(order.remaining_notional), capacity, held_notional)
        if fill <= 1e-8:
            continue
        quantity = fill / float(price)
        shares[order.symbol] = max(held_quantity - quantity, 0.0)
        if shares[order.symbol] <= 1e-10:
            shares.pop(order.symbol, None)
        cost = _trade_fee(fill, side="sell", cost_rate=cost_rate, fee_model=trade_fee_model)
        cash_ref["cash"] = float(cash_ref.get("cash", 0.0)) + fill - cost
        _update_nav_order(order, trade_date, fill)
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


def _update_nav_order(order: _NavOrder, trade_date: pd.Timestamp, fill: float) -> None:
    order.filled_notional += float(fill)
    order.remaining_notional = max(float(order.remaining_notional) - float(fill), 0.0)
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
) -> None:
    fill_rows.append(
        {
            "rebalance_date": _format_date(order.rebalance_date),
            "entry_date": _format_date(order.entry_date),
            "trade_date": _format_date(trade_date),
            "day_number": int(trade_idx - order.start_idx + 1),
            "side": order.side,
            "symbol": order.symbol,
            "remaining_before_notional": float(order.remaining_notional + filled_notional),
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
        if order.remaining_notional <= 1e-8:
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
    status = order.status or ("filled" if order.remaining_notional <= 1e-8 else "open")
    order_rows.append(
        {
            "rebalance_date": _format_date(order.rebalance_date),
            "entry_date": _format_date(order.entry_date),
            "side": order.side,
            "symbol": order.symbol,
            "requested_notional": float(order.requested_notional),
            "filled_notional": float(order.filled_notional),
            "unfilled_notional": float(max(order.remaining_notional, 0.0)),
            "fill_ratio": float(order.filled_notional / order.requested_notional)
            if order.requested_notional > 0
            else np.nan,
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


def _build_targets_by_rebalance(
    positions: pd.DataFrame,
) -> list[tuple[pd.Timestamp, dict[str, Any]]]:
    grouped = []
    for rebalance_date, group in positions.groupby("rebalance_date", sort=True):
        entry_date = pd.to_datetime(group["entry_date"].iloc[0])
        weights = (
            group.groupby("symbol")["weight"]
            .sum()
            .astype(float)
            .loc[lambda series: series > 0]
            .to_dict()
        )
        grouped.append(
            (pd.to_datetime(rebalance_date), {"entry_date": entry_date, "weights": weights})
        )
    return grouped


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
