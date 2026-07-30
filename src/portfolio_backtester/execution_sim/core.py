"""Order-level capacity execution simulation for rebalance targets."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..execution import DetailedTradeFeeModel
from .capacity import (
    _positions_value,
    _refresh_last_prices,
)
from .config import (
    ExecutionSimConfig,
    required_execution_sim_columns,
)
from .models import (
    _AdjustedNavLedger,
    _AdjustedNavPlan,
    _ExecutionTables,
    _NavOrder,
    _OrderSink,
)
from .orders import (
    _append_nav_order_row,
    _build_nav_orders_for_target,
    _build_targets_by_rebalance,
    _cash_weight_breakdown,
    _execute_buy_orders,
    _execute_nav_orders_for_day,
    _execute_sell_orders,
    _finalize_open_nav_orders,
    _nav_order_is_complete,
    _nav_order_should_abort_buy,
    _rebalance_ideal_target,
    _target_cash_notional,
)
from .reporting import (
    _empty_adjusted_nav_result,
    _empty_result,
    _executed_daily_columns,
    _fill_columns,
    _format_date,
    _nav_fill_columns,
    _nav_order_columns,
    _order_columns,
    _summarize_adjusted_nav,
    _summarize_orders,
)
from .results import (
    ExecutionAdjustedNavResult,
    ExecutionSimResult,
)

TradeFeeModel = DetailedTradeFeeModel


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
        if _nav_order_is_complete(order):
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
