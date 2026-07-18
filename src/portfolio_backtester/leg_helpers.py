from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import pandas as pd

from .execution import CostModel, SlippageModel
from .portfolio_weights import (
    build_position_weights,
    clean_position_weights,
    limit_weight_turnover,
)
from .pricing import slippage_pricing_row
from .selection_controls import TargetWeightPolicy
from .turnover import (
    RebalanceTurnoverReport,
    build_rebalance_turnover_report,
    turnover_from_trade_weights,
)
from .types import BacktestLegResult, BacktestPositionState


@dataclass(frozen=True)
class _TargetWeightsAndExit:
    target_weights: pd.Series
    requested_weights: pd.Series


def _build_target_weights_and_exit(
    *,
    day: pd.DataFrame,
    holdings: list[str],
    pred_col: str,
    side: Literal["long", "short"],
    weighting_mode: str,
    weighting_liquidity_col: str,
    previous: BacktestPositionState,
    max_turnover_per_rebalance: float | None,
    selection_min_score: float | None,
    target_weight_policy: TargetWeightPolicy,
    target_slot_count: int,
    preserve_gross_exposure: bool,
) -> _TargetWeightsAndExit:
    if not holdings:
        empty = pd.Series(dtype=float)
        return _TargetWeightsAndExit(empty, empty)
    target_weights = build_position_weights(
        day,
        holdings,
        pred_col,
        side=side,
        weighting=weighting_mode,
        liquidity_col=weighting_liquidity_col,
        target_weight_policy=target_weight_policy,
        target_slot_count=target_slot_count,
    )
    requested_weights = limit_weight_turnover(
        previous.weights,
        target_weights,
        max_turnover_per_rebalance,
        preserve_gross_exposure=preserve_gross_exposure,
    )
    if selection_min_score is not None and not preserve_gross_exposure:
        requested_weights = clean_position_weights(
            requested_weights.reindex(holdings).dropna(),
            preserve_gross_exposure=False,
        )
    return _TargetWeightsAndExit(
        target_weights=target_weights,
        requested_weights=requested_weights,
    )


def _next_position_state(
    leg: BacktestLegResult,
    *,
    entry_date: pd.Timestamp,
) -> BacktestPositionState:
    initial_cash = not leg.holdings and leg.is_initial
    has_recorded_target = bool(leg.target_holdings) or not leg.is_initial
    return BacktestPositionState(
        holdings=None if initial_cash else set(leg.holdings),
        weights=None if initial_cash else leg.weights,
        entry_date=None if initial_cash else entry_date,
        entry_prices=None if initial_cash else leg.entry_prices,
        target_holdings=set(leg.target_holdings) if has_recorded_target else None,
        target_weights=leg.target_weights if has_recorded_target else None,
    )


def _compute_trade_summary(
    prev_weights: pd.Series | None,
    prev_prices: pd.Series | None,
    prev_date: pd.Timestamp | None,
    target_weights: pd.Series,
    entry_date: pd.Timestamp,
    *,
    price_table: pd.DataFrame,
    preserve_gross_exposure: bool = False,
) -> tuple[float, float, float, pd.Series]:
    target_clean = clean_position_weights(
        target_weights,
        preserve_gross_exposure=preserve_gross_exposure,
    )

    if prev_weights is None:
        if target_clean.empty:
            return 0.0, 0.0, 0.0, pd.Series(dtype=float)
        trade_weights = target_clean.copy()
        turnover = turnover_from_trade_weights(trade_weights, is_initial=True)
        return (
            turnover.one_way_turnover,
            turnover.buy_weight,
            turnover.sell_weight,
            trade_weights,
        )
    if prev_weights.empty:
        if target_clean.empty:
            return 0.0, 0.0, 0.0, pd.Series(dtype=float)
        trade_weights = target_clean.copy()
        turnover = turnover_from_trade_weights(trade_weights)
        return (
            turnover.one_way_turnover,
            turnover.buy_weight,
            turnover.sell_weight,
            trade_weights,
        )

    prev_clean = clean_position_weights(
        prev_weights,
        preserve_gross_exposure=preserve_gross_exposure,
    )
    drift_weights = _drift_previous_weights(
        prev_clean,
        prev_prices,
        prev_date,
        entry_date,
        price_table=price_table,
        preserve_gross_exposure=preserve_gross_exposure,
    )
    all_ids = drift_weights.index.union(target_clean.index)
    drift_aligned = drift_weights.reindex(all_ids).fillna(0.0)
    target_aligned = target_clean.reindex(all_ids).fillna(0.0)
    trade_weights = target_aligned - drift_aligned
    turnover = turnover_from_trade_weights(trade_weights)
    return (
        turnover.one_way_turnover,
        turnover.buy_weight,
        turnover.sell_weight,
        trade_weights,
    )


def _drift_previous_weights(
    prev_clean: pd.Series,
    prev_prices: pd.Series | None,
    prev_date: pd.Timestamp | None,
    entry_date: pd.Timestamp,
    *,
    price_table: pd.DataFrame,
    preserve_gross_exposure: bool,
) -> pd.Series:
    if prev_prices is None or prev_date is None:
        return prev_clean
    prev_prices_valid = cast(pd.Series, prev_prices.reindex(prev_clean.index))
    prev_prices_valid = cast(pd.Series, prev_prices_valid[prev_prices_valid.notna()])
    if prev_prices_valid.empty or entry_date not in price_table.index:
        return prev_clean
    prev_clean = prev_clean.reindex(prev_prices_valid.index).dropna()
    current_prices = cast(pd.Series, price_table.loc[entry_date, prev_prices_valid.index])
    valid_prev = current_prices.notna()
    prev_prices_valid = cast(pd.Series, prev_prices_valid[valid_prev])
    current_prices = current_prices[valid_prev]
    prev_clean = prev_clean.reindex(prev_prices_valid.index).dropna()
    if prev_prices_valid.empty or prev_clean.empty:
        return prev_clean
    drift_values = prev_clean * (current_prices / prev_prices_valid)
    drift_sum = float(drift_values.sum())
    if drift_sum <= 0:
        return prev_clean
    if not preserve_gross_exposure:
        return clean_position_weights(drift_values, preserve_gross_exposure=False)
    cash_weight = max(0.0, 1.0 - float(prev_clean.sum()))
    nav = cash_weight + drift_sum
    if nav <= 0:
        return prev_clean
    return clean_position_weights(
        drift_values / nav,
        preserve_gross_exposure=True,
    )


def _build_leg_turnover_report(
    previous: BacktestPositionState,
    target_weights: pd.Series,
    trade_weights: pd.Series,
) -> RebalanceTurnoverReport:
    previous_holdings = (
        previous.target_holdings if previous.target_holdings is not None else previous.holdings
    )
    previous_target_weights = (
        previous.target_weights if previous.target_weights is not None else previous.weights
    )
    return build_rebalance_turnover_report(
        previous_holdings=previous_holdings,
        target_holdings=target_weights[target_weights > 1e-12].index,
        previous_target_weights=previous_target_weights,
        target_weights=target_weights,
        pretrade_trade_weights=trade_weights,
    )


def _build_backtest_leg_result(
    *,
    holdings: list[str],
    target_weights: pd.Series,
    weights: pd.Series,
    entry_prices: pd.Series,
    exit_prices: pd.Series,
    period_exit_idx: int,
    entry_idx: int,
    entry_date: pd.Timestamp,
    trade_dates: list[pd.Timestamp],
    entry_price_table: pd.DataFrame,
    side: Literal["long", "short"],
    previous: BacktestPositionState,
    cost_model: CostModel,
    slippage_model: SlippageModel,
    amount_tables: dict[str, pd.DataFrame],
    preserve_gross_exposure: bool = False,
) -> BacktestLegResult:
    period_returns = (exit_prices / entry_prices) - 1.0
    gross = float((period_returns * weights.reindex(period_returns.index)).sum())
    if side == "short":
        gross = -gross

    turnover, entry_turnover, exit_turnover, trade_weights = _compute_trade_summary(
        previous.weights,
        previous.entry_prices,
        previous.entry_date,
        weights,
        entry_date,
        price_table=entry_price_table,
        preserve_gross_exposure=preserve_gross_exposure,
    )
    turnover_breakdown = turnover_from_trade_weights(
        trade_weights,
        is_initial=previous.weights is None,
    )
    turnover_report = _build_leg_turnover_report(previous, target_weights, trade_weights)
    fee_cost = cost_model.cost(
        turnover,
        is_initial=previous.weights is None,
        side=side,
        entry_turnover=entry_turnover,
        exit_turnover=exit_turnover,
        holding_days=int(period_exit_idx - entry_idx),
        gross_exposure=float(weights.abs().sum()),
    )
    slippage_cost = slippage_model.cost(
        trade_weights,
        pricing_row=slippage_pricing_row(
            slippage_model=slippage_model,
            amount_tables=amount_tables,
            entry_date=entry_date,
        ),
        is_initial=previous.weights is None,
        side=side,
    )
    return BacktestLegResult(
        holdings=holdings,
        weights=weights,
        entry_prices=entry_prices,
        exit_idx=period_exit_idx,
        exit_date=trade_dates[period_exit_idx],
        gross=gross,
        turnover=turnover,
        fee_cost=fee_cost,
        slippage_cost=slippage_cost,
        buy_turnover=turnover_breakdown.buy_weight,
        sell_turnover=turnover_breakdown.sell_weight,
        gross_traded_weight=turnover_breakdown.gross_traded_weight,
        half_l1_turnover=turnover_breakdown.half_l1_turnover,
        is_initial=turnover_breakdown.is_initial,
        target_name_turnover=turnover_report.target_name_turnover,
        target_entered_names=turnover_report.target_entered_names,
        target_exited_names=turnover_report.target_exited_names,
        target_overlap_names=turnover_report.target_overlap_names,
        target_weight_full_l1=turnover_report.target_weight_full_l1,
        target_weight_half_l1=turnover_report.target_weight_half_l1,
        pretrade_demand_buy=turnover_report.pretrade_demand_buy,
        pretrade_demand_sell=turnover_report.pretrade_demand_sell,
        pretrade_demand_full_l1=turnover_report.pretrade_demand_full_l1,
        pretrade_demand_half_l1=turnover_report.pretrade_demand_half_l1,
        executed_buy=turnover_report.executed_buy,
        executed_sell=turnover_report.executed_sell,
        executed_gross=turnover_report.executed_gross,
        executed_full_l1=turnover_report.executed_full_l1,
        executed_half_l1=turnover_report.executed_half_l1,
        executed_cost=turnover_report.executed_cost,
        target_holdings=tuple(str(symbol) for symbol in target_weights.index),
        target_weights=target_weights.copy(),
        target_gross_exposure=float(target_weights.abs().sum()),
        target_cash_weight=max(0.0, 1.0 - float(target_weights.abs().sum())),
        modeled_gross_exposure=float(weights.abs().sum()),
        modeled_cash_weight=max(0.0, 1.0 - float(weights.abs().sum())),
    )
