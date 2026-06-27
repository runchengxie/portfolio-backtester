from __future__ import annotations

from typing import Literal

import pandas as pd
from market_data_platform.symbols import canonicalize_symbol_columns

from ..contracts.backtest import assert_backtest_pricing_frame
from .execution import (
    BpsCostModel,
    EntryPolicy,
    ExecutionModel,
    ExitPolicy,
    NoSlippageModel,
    ParticipationSlippageModel,
    SelectionConstraints,
    SlippageModel,
)
from .types import BacktestExecutionContext, BacktestPricingContext


def normalize_backtest_frame(
    frame: pd.DataFrame | None,
    *,
    context: str,
) -> pd.DataFrame | None:
    if frame is None or frame.empty:
        return frame
    normalized = canonicalize_symbol_columns(frame, context=context)
    normalized = normalized.copy()
    normalized["trade_date"] = pd.to_datetime(normalized["trade_date"]).dt.normalize()
    return normalized


def resolve_backtest_execution_context(
    *,
    execution: ExecutionModel | None,
    exit_price_policy: Literal["strict", "ffill", "delay"],
    exit_fallback_policy: Literal["ffill", "none"],
    price_col: str,
    cost_bps: float,
) -> BacktestExecutionContext:
    if execution is not None:
        return BacktestExecutionContext(
            exit_policy=execution.exit_policy,
            cost_model=execution.cost_model,
            slippage_model=execution.slippage_model,
            entry_policy=execution.entry_policy,
            selection_constraints=execution.selection_constraints,
            calendar=execution.calendar,
            open_dates=execution.calendar_open_dates,
            closed_dates=execution.calendar_closed_dates,
        )

    if exit_price_policy not in {"strict", "ffill", "delay"}:
        raise ValueError("exit_price_policy must be one of: strict, ffill, delay.")
    if exit_fallback_policy not in {"ffill", "none"}:
        raise ValueError("exit_fallback_policy must be one of: ffill, none.")
    return BacktestExecutionContext(
        exit_policy=ExitPolicy(exit_price_policy, exit_fallback_policy, price_col),
        cost_model=BpsCostModel(cost_bps),
        slippage_model=NoSlippageModel(),
        entry_policy=EntryPolicy(price_col),
        selection_constraints=SelectionConstraints(),
        calendar="market",
        open_dates=(),
        closed_dates=(),
    )


def prepare_backtest_pricing_context(
    *,
    data: pd.DataFrame | None,
    pricing_data: pd.DataFrame | None,
    entry_policy: EntryPolicy,
    exit_policy: ExitPolicy,
    selection_constraints: SelectionConstraints,
    slippage_model: SlippageModel,
    tradable_col: str | None,
) -> BacktestPricingContext | None:
    pricing_source = pricing_data if pricing_data is not None else data
    if pricing_source is None or pricing_source.empty:
        return None

    entry_price_col = entry_policy.price_col
    exit_price_col = exit_policy.price_col
    amount_columns: list[str] = []
    if selection_constraints.min_amount is not None:
        amount_columns.append(selection_constraints.amount_col)
    if isinstance(slippage_model, ParticipationSlippageModel):
        amount_columns.append(slippage_model.amount_col)
    assert_backtest_pricing_frame(
        pricing_source,
        entry_price_col=entry_price_col,
        exit_price_col=exit_price_col,
        amount_columns=amount_columns,
        tradable_col=tradable_col,
    )

    pricing_source = pricing_source.drop_duplicates(subset=["trade_date", "symbol"]).copy()
    trade_dates = [
        pd.Timestamp(date).normalize() for date in sorted(pricing_source["trade_date"].unique())
    ]
    if len(trade_dates) < 2:
        return None
    date_to_idx = {date: idx for idx, date in enumerate(trade_dates)}
    entry_price_table = pricing_source.pivot(
        index="trade_date", columns="symbol", values=entry_price_col
    )
    exit_price_table = pricing_source.pivot(
        index="trade_date", columns="symbol", values=exit_price_col
    )
    day_groups: dict[pd.Timestamp, pd.DataFrame] = {}
    if data is not None:
        for date, group in data.groupby("trade_date", sort=False):
            day_groups[date] = group
    tradable_table = None
    if tradable_col and tradable_col in pricing_source.columns:
        tradable_table = pricing_source.pivot(
            index="trade_date", columns="symbol", values=tradable_col
        )
        tradable_table = tradable_table.fillna(False).astype(bool)
    amount_tables: dict[str, pd.DataFrame] = {}
    for amount_col in sorted(set(amount_columns)):
        amount_tables[amount_col] = pricing_source.pivot(
            index="trade_date", columns="symbol", values=amount_col
        )

    return BacktestPricingContext(
        trade_dates=trade_dates,
        date_to_idx=date_to_idx,
        entry_price_table=entry_price_table,
        exit_price_table=exit_price_table,
        day_groups=day_groups,
        tradable_table=tradable_table,
        amount_tables=amount_tables,
    )


def slippage_pricing_row(
    *,
    slippage_model: SlippageModel,
    amount_tables: dict[str, pd.DataFrame],
    entry_date: pd.Timestamp,
) -> pd.Series | None:
    if not isinstance(slippage_model, ParticipationSlippageModel):
        return None
    return amount_tables[slippage_model.amount_col].loc[entry_date]
