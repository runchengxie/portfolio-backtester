from __future__ import annotations

from typing import Literal, cast

import pandas as pd

from .execution import CostModel, SlippageModel
from .portfolio_weights import normalize_position_weights
from .pricing import slippage_pricing_row
from .turnover import turnover_from_trade_weights
from .types import BacktestLegResult, BacktestPositionState


def _compute_trade_summary(
    prev_weights: pd.Series | None,
    prev_prices: pd.Series | None,
    prev_date: pd.Timestamp | None,
    target_weights: pd.Series,
    entry_date: pd.Timestamp,
    *,
    price_table: pd.DataFrame,
) -> tuple[float, float, float, pd.Series]:
    if target_weights is None or target_weights.empty:
        return 0.0, 0.0, 0.0, pd.Series(dtype=float)

    target_clean = normalize_position_weights(target_weights)
    if target_clean.empty:
        return 0.0, 0.0, 0.0, pd.Series(dtype=float)

    if prev_weights is None or prev_weights.empty:
        trade_weights = target_clean.copy()
        turnover = turnover_from_trade_weights(trade_weights, is_initial=True)
        return (
            turnover.one_way_turnover,
            turnover.buy_weight,
            turnover.sell_weight,
            trade_weights,
        )

    prev_clean = normalize_position_weights(prev_weights)
    drift_weights = _drift_previous_weights(
        prev_clean,
        prev_prices,
        prev_date,
        entry_date,
        price_table=price_table,
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
    drift = prev_clean * (current_prices / prev_prices_valid)
    drift_sum = float(drift.sum())
    if drift_sum <= 0:
        return prev_clean
    return normalize_position_weights(drift)


def _build_backtest_leg_result(
    *,
    holdings: list[str],
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
    )
    turnover_breakdown = turnover_from_trade_weights(
        trade_weights,
        is_initial=previous.weights is None,
    )
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
    )
