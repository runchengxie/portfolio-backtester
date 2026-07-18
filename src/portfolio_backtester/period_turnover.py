"""Period-level aggregation for leg returns, costs, and turnover diagnostics."""

from __future__ import annotations

import pandas as pd

from .types import BacktestLegResult, BacktestPeriodResult


def _optional_sum(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return float(left + right)


def _merged_names(left: tuple[str, ...], right: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted(set(left) | set(right)))


def period_result_from_leg(leg: BacktestLegResult) -> BacktestPeriodResult:
    """Promote one long-only leg without losing turnover layers."""

    total_cost = leg.fee_cost + leg.slippage_cost
    return BacktestPeriodResult(
        gross=leg.gross,
        net=leg.gross - total_cost,
        turnover=leg.turnover,
        fee_cost=leg.fee_cost,
        slippage_cost=leg.slippage_cost,
        total_cost=total_cost,
        exit_idx=leg.exit_idx,
        exit_date=leg.exit_date,
        target_name_turnover=leg.target_name_turnover,
        target_entered_names=leg.target_entered_names,
        target_exited_names=leg.target_exited_names,
        target_overlap_names=leg.target_overlap_names,
        target_weight_full_l1=leg.target_weight_full_l1,
        target_weight_half_l1=leg.target_weight_half_l1,
        pretrade_demand_buy=leg.pretrade_demand_buy,
        pretrade_demand_sell=leg.pretrade_demand_sell,
        pretrade_demand_full_l1=leg.pretrade_demand_full_l1,
        pretrade_demand_half_l1=leg.pretrade_demand_half_l1,
        executed_buy=leg.executed_buy,
        executed_sell=leg.executed_sell,
        executed_gross=leg.executed_gross,
        executed_full_l1=leg.executed_full_l1,
        executed_half_l1=leg.executed_half_l1,
        executed_cost=leg.executed_cost,
        is_initial_build=leg.is_initial,
    )


def period_result_from_legs(
    long_leg: BacktestLegResult,
    short_leg: BacktestLegResult,
    *,
    trade_dates: list[pd.Timestamp],
) -> BacktestPeriodResult:
    """Aggregate independently normalized long and short leg diagnostics."""

    exit_idx = max(long_leg.exit_idx, short_leg.exit_idx)
    fee_cost = long_leg.fee_cost + short_leg.fee_cost
    slippage_cost = long_leg.slippage_cost + short_leg.slippage_cost
    total_cost = fee_cost + slippage_cost
    gross = long_leg.gross + short_leg.gross
    return BacktestPeriodResult(
        gross=gross,
        net=gross - total_cost,
        turnover=long_leg.turnover + short_leg.turnover,
        fee_cost=fee_cost,
        slippage_cost=slippage_cost,
        total_cost=total_cost,
        exit_idx=exit_idx,
        exit_date=trade_dates[exit_idx],
        target_name_turnover=_optional_sum(
            long_leg.target_name_turnover, short_leg.target_name_turnover
        ),
        target_entered_names=_merged_names(
            long_leg.target_entered_names, short_leg.target_entered_names
        ),
        target_exited_names=_merged_names(
            long_leg.target_exited_names, short_leg.target_exited_names
        ),
        target_overlap_names=_merged_names(
            long_leg.target_overlap_names, short_leg.target_overlap_names
        ),
        target_weight_full_l1=_optional_sum(
            long_leg.target_weight_full_l1, short_leg.target_weight_full_l1
        ),
        target_weight_half_l1=_optional_sum(
            long_leg.target_weight_half_l1, short_leg.target_weight_half_l1
        ),
        pretrade_demand_buy=_optional_sum(
            long_leg.pretrade_demand_buy, short_leg.pretrade_demand_buy
        ),
        pretrade_demand_sell=_optional_sum(
            long_leg.pretrade_demand_sell, short_leg.pretrade_demand_sell
        ),
        pretrade_demand_full_l1=_optional_sum(
            long_leg.pretrade_demand_full_l1, short_leg.pretrade_demand_full_l1
        ),
        pretrade_demand_half_l1=_optional_sum(
            long_leg.pretrade_demand_half_l1, short_leg.pretrade_demand_half_l1
        ),
        executed_buy=_optional_sum(long_leg.executed_buy, short_leg.executed_buy),
        executed_sell=_optional_sum(long_leg.executed_sell, short_leg.executed_sell),
        executed_gross=_optional_sum(long_leg.executed_gross, short_leg.executed_gross),
        executed_full_l1=_optional_sum(long_leg.executed_full_l1, short_leg.executed_full_l1),
        executed_half_l1=_optional_sum(long_leg.executed_half_l1, short_leg.executed_half_l1),
        executed_cost=_optional_sum(long_leg.executed_cost, short_leg.executed_cost),
        is_initial_build=long_leg.is_initial and short_leg.is_initial,
    )


def period_turnover_fields(
    result: BacktestPeriodResult,
) -> dict[str, float | bool | int | tuple[str, ...] | None]:
    """Return the flat period-output contract for turnover reporting."""

    return {
        "target_name_turnover": result.target_name_turnover,
        "target_entered_names": result.target_entered_names,
        "target_exited_names": result.target_exited_names,
        "target_overlap_names": result.target_overlap_names,
        "target_entered_count": len(result.target_entered_names),
        "target_exited_count": len(result.target_exited_names),
        "target_overlap_count": len(result.target_overlap_names),
        "target_weight_full_l1": result.target_weight_full_l1,
        "target_weight_half_l1": result.target_weight_half_l1,
        "pretrade_demand_buy": result.pretrade_demand_buy,
        "pretrade_demand_sell": result.pretrade_demand_sell,
        "pretrade_demand_full_l1": result.pretrade_demand_full_l1,
        "pretrade_demand_half_l1": result.pretrade_demand_half_l1,
        "modeled_fee_cost": result.fee_cost,
        "modeled_slippage_cost": result.slippage_cost,
        "modeled_total_cost": result.total_cost,
        "executed_buy": result.executed_buy,
        "executed_sell": result.executed_sell,
        "executed_gross": result.executed_gross,
        "executed_full_l1": result.executed_full_l1,
        "executed_half_l1": result.executed_half_l1,
        "executed_cost": result.executed_cost,
        "execution_data_available": result.executed_full_l1 is not None,
        "is_initial_build": result.is_initial_build,
    }
