"""Research-only staggered cohorts: T-close signal, T+1-open entry, h-day hold."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from .staggered_cohort_execution_records import append_buy_order, append_sell_order
from .staggered_cohort_execution_reporting import (
    append_day_outputs,
    build_result,
    check_conservation,
    check_cost_identity,
    finalize_generations,
)
from .staggered_cohort_execution_state import (
    EPSILON,
    Cohort,
    DayFlow,
    Generation,
    Ledger,
    Position,
    StaggeredCohortExecutionConfig,
    StaggeredCohortExecutionResult,
    ledger_nav,
)
from .staggered_cohort_inputs import (
    StaggeredTarget,
    prepare_staggered_pricing,
    prepare_staggered_targets,
)


def simulate_staggered_cohort_execution(
    signals: pd.DataFrame,
    pricing: pd.DataFrame,
    config: StaggeredCohortExecutionConfig,
    *,
    trade_calendar: pd.DataFrame | pd.DatetimeIndex,
) -> StaggeredCohortExecutionResult:
    """Run a PIT-safe open-to-open cohort ledger without forcing terminal liquidation."""

    prices, trade_dates = prepare_staggered_pricing(
        pricing,
        trade_calendar,
        valuation_price_col=config.valuation_price_col,
    )
    targets = prepare_staggered_targets(
        signals,
        trade_dates,
        horizon_days=config.horizon_days,
        top_n=config.top_n,
        score_col=config.score_col,
        signal_date_col=config.signal_date_col,
        available_at_col=config.available_at_col,
    )
    return _run_ledger(prices, trade_dates, targets, config)


def _run_ledger(
    prices: pd.DataFrame,
    trade_dates: list[pd.Timestamp],
    targets: list[StaggeredTarget],
    config: StaggeredCohortExecutionConfig,
) -> StaggeredCohortExecutionResult:
    target_by_entry = {target.entry_date: target for target in targets}
    ledger = _initial_ledger(config)
    previous_nav = float(config.initial_capital)
    for trade_idx in range(targets[0].entry_idx, len(trade_dates)):
        trade_date = trade_dates[trade_idx]
        _mark_positions(ledger, prices, trade_date, config.valuation_price_col)
        pretrade_nav = ledger_nav(ledger)
        flow = _sell_due_positions(ledger, prices, trade_date, trade_idx, config)
        target = target_by_entry.get(trade_date)
        if target is not None:
            _enter_target(ledger, prices, target, config, flow)
        finalize_generations(ledger, trade_date, trade_idx)
        net_nav = ledger_nav(ledger)
        check_conservation(ledger, net_nav)
        check_cost_identity(pretrade_nav, net_nav, flow.transaction_cost)
        append_day_outputs(ledger, trade_date, trade_idx, pretrade_nav, previous_nav, net_nav, flow)
        previous_nav = net_nav
    return build_result(ledger, config, trade_dates[-1])


def _initial_ledger(config: StaggeredCohortExecutionConfig) -> Ledger:
    sleeve_cash = float(config.initial_capital) / config.horizon_days
    cohorts = {idx: Cohort(cohort_id=idx, cash=sleeve_cash) for idx in range(config.horizon_days)}
    return Ledger(cohorts=cohorts)


def _price_row(prices: pd.DataFrame, trade_date: pd.Timestamp, symbol: str) -> pd.Series | None:
    key = (trade_date, symbol)
    if key not in prices.index:
        return None
    return cast(pd.Series, prices.loc[key])


def _positive_number(value: object) -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric) or not np.isfinite(float(numeric)) or float(numeric) <= 0:
        return None
    return float(numeric)


def _suspension_state(value: object) -> bool | None:
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if text in {"true", "t", "yes", "y", "1", "1.0"}:
        return True
    if text in {"false", "f", "no", "n", "0", "0.0"}:
        return False
    return None


def _block_reason(row: pd.Series | None, *, side: str) -> str | None:
    if row is None:
        return "missing_price_row"
    suspended = _suspension_state(row.get("is_suspended"))
    if suspended is None:
        return "unknown_suspension"
    if suspended:
        return "suspended"
    open_price = _positive_number(row.get("open"))
    if open_price is None:
        return "missing_open"
    limit_col = "up_limit" if side == "buy" else "down_limit"
    limit_price = _positive_number(row.get(limit_col))
    if limit_price is None:
        return f"missing_{limit_col}"
    tolerance = max(abs(limit_price) * 1e-8, 1e-8)
    if side == "buy" and open_price >= limit_price - tolerance:
        return "limit_up_open"
    if side == "sell" and open_price <= limit_price + tolerance:
        return "limit_down_open"
    return None


def _valuation_price(row: pd.Series | None, *, column: str) -> float | None:
    return None if row is None else _positive_number(row.get(column))


def _execution_block_reason(
    row: pd.Series | None,
    *,
    side: str,
    valuation_price_col: str,
) -> str | None:
    raw_reason = _block_reason(row, side=side)
    if raw_reason is not None:
        return raw_reason
    if _valuation_price(row, column=valuation_price_col) is None:
        return "missing_valuation_price"
    return None


def _mark_positions(
    ledger: Ledger,
    prices: pd.DataFrame,
    trade_date: pd.Timestamp,
    valuation_price_col: str,
) -> None:
    for cohort in ledger.cohorts.values():
        for position in cohort.positions.values():
            row = _price_row(prices, trade_date, position.symbol)
            price = _valuation_price(row, column=valuation_price_col)
            if price is not None:
                position.last_price = price


def _sell_due_positions(
    ledger: Ledger,
    prices: pd.DataFrame,
    trade_date: pd.Timestamp,
    trade_idx: int,
    config: StaggeredCohortExecutionConfig,
) -> DayFlow:
    flow = DayFlow()
    for cohort in ledger.cohorts.values():
        due = [pos for pos in cohort.positions.values() if pos.planned_exit_idx <= trade_idx]
        for position in sorted(due, key=lambda item: item.position_id):
            _sell_position(ledger, cohort, position, prices, trade_date, config, flow)
    return flow


def _sell_position(
    ledger: Ledger,
    cohort: Cohort,
    position: Position,
    prices: pd.DataFrame,
    trade_date: pd.Timestamp,
    config: StaggeredCohortExecutionConfig,
    flow: DayFlow,
) -> None:
    row = _price_row(prices, trade_date, position.symbol)
    reason = _execution_block_reason(
        row,
        side="sell",
        valuation_price_col=config.valuation_price_col,
    )
    requested = float(position.shares * position.last_price)
    if reason is not None:
        _record_blocked_sell(ledger, position, row, trade_date, requested, reason, config, flow)
        return
    valuation_price = cast(float, _valuation_price(row, column=config.valuation_price_col))
    filled = float(position.shares * valuation_price)
    cost = filled * config.single_side_cost_bps / 10_000.0
    cohort.cash += filled - cost
    generation = ledger.generations[position.generation_id]
    generation.gross_proceeds += filled
    generation.sell_cost += cost
    flow.traded_notional += filled
    flow.transaction_cost += cost
    append_sell_order(
        ledger,
        trade_date,
        position,
        row,
        filled,
        filled,
        cost,
        None,
        valuation_price_col=config.valuation_price_col,
    )
    del cohort.positions[position.position_id]


def _record_blocked_sell(
    ledger: Ledger,
    position: Position,
    row: pd.Series | None,
    trade_date: pd.Timestamp,
    requested: float,
    reason: str,
    config: StaggeredCohortExecutionConfig,
    flow: DayFlow,
) -> None:
    position.carry_days += 1
    flow.blocked_sell_notional += requested
    if reason == "suspended":
        flow.suspended_sell_notional += requested
    append_sell_order(
        ledger,
        trade_date,
        position,
        row,
        requested,
        0.0,
        0.0,
        reason,
        valuation_price_col=config.valuation_price_col,
    )


def _enter_target(
    ledger: Ledger,
    prices: pd.DataFrame,
    target: StaggeredTarget,
    config: StaggeredCohortExecutionConfig,
    flow: DayFlow,
) -> None:
    cohort = ledger.cohorts[target.cohort_id]
    generation_id = f"c{target.cohort_id}:{target.entry_date:%Y%m%d}"
    budget = max(float(cohort.cash), 0.0)
    generation = Generation(
        generation_id=generation_id,
        cohort_id=target.cohort_id,
        signal_date=target.signal_date,
        entry_date=target.entry_date,
        planned_exit_date=target.planned_exit_date,
        planned_exit_idx=target.planned_exit_idx,
        selected_symbols=target.symbols,
        allocation_budget=budget,
    )
    ledger.generations[generation_id] = generation
    slot_budget = budget / config.top_n
    for symbol, score in zip(target.symbols, target.scores, strict=True):
        _enter_symbol(
            ledger, cohort, generation, prices, target, symbol, score, slot_budget, config, flow
        )


def _enter_symbol(
    ledger: Ledger,
    cohort: Cohort,
    generation: Generation,
    prices: pd.DataFrame,
    target: StaggeredTarget,
    symbol: str,
    score: float,
    slot_budget: float,
    config: StaggeredCohortExecutionConfig,
    flow: DayFlow,
) -> None:
    cost_rate = config.single_side_cost_bps / 10_000.0
    requested = slot_budget / (1.0 + cost_rate) if slot_budget > 0 else 0.0
    row = _price_row(prices, target.entry_date, symbol)
    reason = _buy_block_reason(row, slot_budget, config.valuation_price_col)
    if reason is not None:
        _record_blocked_buy(
            ledger,
            generation,
            target,
            symbol,
            score,
            row,
            slot_budget,
            requested,
            reason,
            config,
            flow,
        )
        return
    valuation_price = cast(float, _valuation_price(row, column=config.valuation_price_col))
    cost = requested * cost_rate
    cohort.cash -= requested + cost
    if abs(cohort.cash) <= EPSILON:
        cohort.cash = 0.0
    position_id = f"{generation.generation_id}:{symbol}"
    cohort.positions[position_id] = Position(
        position_id=position_id,
        generation_id=generation.generation_id,
        cohort_id=cohort.cohort_id,
        symbol=symbol,
        shares=requested / valuation_price,
        entry_date=target.entry_date,
        entry_price=valuation_price,
        entry_notional=requested,
        planned_exit_date=target.planned_exit_date,
        planned_exit_idx=target.planned_exit_idx,
        last_price=valuation_price,
    )
    generation.entry_notional += requested
    generation.buy_cost += cost
    flow.traded_notional += requested
    flow.transaction_cost += cost
    append_buy_order(
        ledger,
        target,
        symbol,
        score,
        row,
        slot_budget,
        requested,
        requested,
        cost,
        None,
        valuation_price_col=config.valuation_price_col,
    )


def _buy_block_reason(
    row: pd.Series | None,
    slot_budget: float,
    valuation_price_col: str,
) -> str | None:
    if slot_budget <= EPSILON:
        return "unfunded_cash"
    return _execution_block_reason(row, side="buy", valuation_price_col=valuation_price_col)


def _record_blocked_buy(
    ledger: Ledger,
    generation: Generation,
    target: StaggeredTarget,
    symbol: str,
    score: float,
    row: pd.Series | None,
    slot_budget: float,
    requested: float,
    reason: str,
    config: StaggeredCohortExecutionConfig,
    flow: DayFlow,
) -> None:
    if reason == "unfunded_cash":
        generation.unfunded_buy_budget += slot_budget
        flow.unfunded_buy_notional += slot_budget
    else:
        generation.blocked_buy_budget += slot_budget
        flow.blocked_buy_notional += slot_budget
        if reason == "suspended":
            flow.suspended_buy_notional += slot_budget
    append_buy_order(
        ledger,
        target,
        symbol,
        score,
        row,
        slot_budget,
        requested,
        0.0,
        0.0,
        reason,
        valuation_price_col=config.valuation_price_col,
    )


__all__ = [
    "StaggeredCohortExecutionConfig",
    "StaggeredCohortExecutionResult",
    "simulate_staggered_cohort_execution",
]
