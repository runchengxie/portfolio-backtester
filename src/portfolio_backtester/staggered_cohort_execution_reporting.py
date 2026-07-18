"""Ledger snapshots, invariants, and result assembly for staggered cohorts."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .staggered_cohort_execution_state import (
    EPSILON,
    DayFlow,
    Ledger,
    StaggeredCohortExecutionConfig,
    StaggeredCohortExecutionResult,
    cohort_nav,
    position_value,
)


def finalize_generations(ledger: Ledger, trade_date: pd.Timestamp, trade_idx: int) -> None:
    live_generation_ids = {
        position.generation_id
        for cohort in ledger.cohorts.values()
        for position in cohort.positions.values()
    }
    for generation in ledger.generations.values():
        if generation.completed_date is not None or trade_idx < generation.planned_exit_idx:
            continue
        if generation.generation_id in live_generation_ids:
            generation.status = "carry"
            continue
        generation.completed_date = trade_date
        if generation.allocation_budget <= EPSILON:
            generation.status = "unfunded"
        elif generation.entry_notional <= EPSILON:
            generation.status = "cash_only"
        else:
            generation.status = "closed"


def append_day_outputs(
    ledger: Ledger,
    trade_date: pd.Timestamp,
    trade_idx: int,
    pretrade_nav: float,
    previous_nav: float,
    net_nav: float,
    flow: DayFlow,
) -> None:
    cash = sum(cohort.cash for cohort in ledger.cohorts.values())
    position_total = sum(
        position_value(position)
        for cohort in ledger.cohorts.values()
        for position in cohort.positions.values()
    )
    denominator = pretrade_nav if pretrade_nav > 0 else np.nan
    ledger.daily.append(
        {
            "trade_date": trade_date,
            "gross_return": pretrade_nav / previous_nav - 1.0,
            "net_return": net_nav / previous_nav - 1.0,
            "gross_nav_before_cost": pretrade_nav,
            "net_nav": net_nav,
            "cash": cash,
            "positions_value": position_total,
            "cash_weight": cash / net_nav,
            "gross_exposure": position_total / net_nav,
            "traded_notional": flow.traded_notional,
            "transaction_cost": flow.transaction_cost,
            "blocked_buy_weight": flow.blocked_buy_notional / denominator,
            "blocked_sell_weight": flow.blocked_sell_notional / denominator,
            "suspended_buy_weight": flow.suspended_buy_notional / denominator,
            "suspended_sell_weight": flow.suspended_sell_notional / denominator,
            "unfunded_buy_weight": flow.unfunded_buy_notional / denominator,
            "position_count": sum(len(cohort.positions) for cohort in ledger.cohorts.values()),
            "is_terminal_complete": pd.NA,
        }
    )
    _append_position_snapshots(ledger, trade_date, trade_idx, net_nav)
    _append_cohort_snapshots(ledger, trade_date, trade_idx, net_nav)


def _append_position_snapshots(
    ledger: Ledger, trade_date: pd.Timestamp, trade_idx: int, total_nav: float
) -> None:
    for cohort in ledger.cohorts.values():
        current_cohort_nav = cohort_nav(cohort)
        for position in sorted(cohort.positions.values(), key=lambda item: item.position_id):
            value = position_value(position)
            ledger.positions.append(
                {
                    "trade_date": trade_date,
                    "cohort_id": cohort.cohort_id,
                    "generation_id": position.generation_id,
                    "symbol": position.symbol,
                    "shares": position.shares,
                    "mark_open": position.last_price,
                    "market_value": value,
                    "portfolio_weight": value / total_nav,
                    "cohort_weight": value / current_cohort_nav,
                    "entry_date": position.entry_date,
                    "planned_exit_date": position.planned_exit_date,
                    "is_carry": trade_idx >= position.planned_exit_idx,
                    "carry_days": position.carry_days,
                }
            )


def _append_cohort_snapshots(
    ledger: Ledger, trade_date: pd.Timestamp, trade_idx: int, total_nav: float
) -> None:
    for cohort in ledger.cohorts.values():
        value = sum(position_value(position) for position in cohort.positions.values())
        nav = cohort.cash + value
        carried = sum(
            position.planned_exit_idx <= trade_idx for position in cohort.positions.values()
        )
        ledger.cohort_daily.append(
            {
                "trade_date": trade_date,
                "cohort_id": cohort.cohort_id,
                "cash": cohort.cash,
                "positions_value": value,
                "cohort_nav": nav,
                "portfolio_weight": nav / total_nav,
                "gross_exposure": value / nav if nav > 0 else np.nan,
                "position_count": len(cohort.positions),
                "carried_position_count": carried,
            }
        )


def check_conservation(ledger: Ledger, net_nav: float) -> None:
    cash = sum(cohort.cash for cohort in ledger.cohorts.values())
    positions = sum(
        position_value(position)
        for cohort in ledger.cohorts.values()
        for position in cohort.positions.values()
    )
    tolerance = max(abs(net_nav) * 1e-10, 1e-8)
    if cash < -tolerance or abs(cash + positions - net_nav) > tolerance:
        raise RuntimeError("staggered cohort cash/position conservation failed")


def check_cost_identity(pretrade_nav: float, net_nav: float, transaction_cost: float) -> None:
    expected = pretrade_nav - transaction_cost
    tolerance = max(abs(expected) * 1e-10, 1e-8)
    if abs(net_nav - expected) > tolerance:
        raise RuntimeError("staggered cohort cost/NAV identity failed")


def _generation_row(generation: Any, live_value: float) -> dict[str, Any]:
    denominator = generation.allocation_budget
    gross_pnl = generation.gross_proceeds - generation.entry_notional
    net_pnl = gross_pnl - generation.buy_cost - generation.sell_cost
    completed = generation.completed_date is not None
    marked_gross_pnl = generation.gross_proceeds + live_value - generation.entry_notional
    marked_net_pnl = marked_gross_pnl - generation.buy_cost - generation.sell_cost
    return {
        "generation_id": generation.generation_id,
        "cohort_id": generation.cohort_id,
        "signal_date": generation.signal_date,
        "entry_date": generation.entry_date,
        "planned_exit_date": generation.planned_exit_date,
        "completed_date": generation.completed_date,
        "selected_symbols": list(generation.selected_symbols),
        "allocation_budget": denominator,
        "entry_notional": generation.entry_notional,
        "blocked_buy_budget": generation.blocked_buy_budget,
        "unfunded_buy_budget": generation.unfunded_buy_budget,
        "gross_proceeds": generation.gross_proceeds,
        "buy_cost": generation.buy_cost,
        "sell_cost": generation.sell_cost,
        "gross_return": gross_pnl / denominator if completed and denominator > EPSILON else np.nan,
        "net_return": net_pnl / denominator if completed and denominator > EPSILON else np.nan,
        "diagnostic_mark_gross_return": (
            marked_gross_pnl / denominator if denominator > EPSILON else np.nan
        ),
        "diagnostic_mark_net_return": (
            marked_net_pnl / denominator if denominator > EPSILON else np.nan
        ),
        "status": generation.status,
    }


def generation_rows(ledger: Ledger) -> list[dict[str, Any]]:
    live_values: dict[str, float] = {}
    for cohort in ledger.cohorts.values():
        for position in cohort.positions.values():
            live_values[position.generation_id] = live_values.get(position.generation_id, 0.0) + (
                position_value(position)
            )
    return [
        _generation_row(generation, live_values.get(generation.generation_id, 0.0))
        for generation in ledger.generations.values()
    ]


def _result_summary(
    daily: pd.DataFrame,
    config: StaggeredCohortExecutionConfig,
    terminal_date: pd.Timestamp,
    terminal_positions: int,
) -> dict[str, Any]:
    terminal_complete = terminal_positions == 0
    marked_nav = float(daily["net_nav"].iloc[-1])
    terminal_position_value = float(daily["positions_value"].iloc[-1])
    return {
        "schema": "research.staggered_cohort_execution.v1",
        "research_only": True,
        "signal_timing": "T_close_signal_T_plus_1_open_entry",
        "holding_period": f"{config.horizon_days}_trading_day_open_to_open",
        "tradability_price": "raw_open",
        "valuation_and_fill_price": config.valuation_price_col,
        "cohort_capital_fraction": 1.0 / config.horizon_days,
        "cost_basis": "actual_filled_notional_times_single_side_bps",
        "blocked_buy_action": "keep_fixed_slot_cash_without_redistribution",
        "blocked_sell_action": "keep_position_and_retry_each_observed_open",
        "missing_valuation_price_policy": "last_observed_valuation_price_carry_forward",
        "trade_calendar_policy": "explicit_authoritative_open_sessions_no_whole_day_gaps",
        "terminal_policy": config.terminal_policy,
        "status": "ok" if terminal_complete else "incomplete_terminal_positions",
        "complete_nav": terminal_complete,
        "final_nav": marked_nav if terminal_complete else None,
        "diagnostic_mark_to_open_nav": marked_nav,
        "terminal_date": terminal_date,
        "terminal_unclosed_position_count": terminal_positions,
        "terminal_unclosed_position_weight": terminal_position_value / marked_nav,
        "no_future_data": True,
        "available_at_policy": "T_date_at_or_after_15:00_and_strictly_before_T_plus_1_open",
        "round_lots_simulated": False,
        "partial_fills_simulated": False,
        "config": {
            "horizon_days": config.horizon_days,
            "top_n": config.top_n,
            "initial_capital": config.initial_capital,
            "single_side_cost_bps": config.single_side_cost_bps,
            "valuation_price_col": config.valuation_price_col,
        },
    }


def build_result(
    ledger: Ledger,
    config: StaggeredCohortExecutionConfig,
    terminal_date: pd.Timestamp,
) -> StaggeredCohortExecutionResult:
    daily = pd.DataFrame(ledger.daily)
    terminal_positions = sum(len(cohort.positions) for cohort in ledger.cohorts.values())
    terminal_complete = terminal_positions == 0
    if not daily.empty:
        daily.loc[daily.index[-1], "is_terminal_complete"] = terminal_complete
    return StaggeredCohortExecutionResult(
        summary=_result_summary(daily, config, terminal_date, terminal_positions),
        daily=daily,
        positions=pd.DataFrame(ledger.positions),
        cohort_daily=pd.DataFrame(ledger.cohort_daily),
        orders=pd.DataFrame(ledger.orders),
        generations=pd.DataFrame(generation_rows(ledger)),
    )
