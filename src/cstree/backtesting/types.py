from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .execution import CostModel, EntryPolicy, ExitPolicy, SelectionConstraints, SlippageModel


@dataclass(frozen=True)
class BacktestExecutionContext:
    exit_policy: ExitPolicy
    cost_model: CostModel
    slippage_model: SlippageModel
    entry_policy: EntryPolicy
    selection_constraints: SelectionConstraints
    calendar: str
    open_dates: tuple
    closed_dates: tuple


@dataclass(frozen=True)
class BacktestPricingContext:
    trade_dates: list[pd.Timestamp]
    date_to_idx: dict[pd.Timestamp, int]
    entry_price_table: pd.DataFrame
    exit_price_table: pd.DataFrame
    day_groups: dict[pd.Timestamp, pd.DataFrame]
    tradable_table: pd.DataFrame | None
    amount_tables: dict[str, pd.DataFrame]


@dataclass(frozen=True)
class BacktestPositionState:
    holdings: set[str] | None = None
    weights: pd.Series | None = None
    entry_date: pd.Timestamp | None = None
    entry_prices: pd.Series | None = None


@dataclass(frozen=True)
class BacktestLegResult:
    holdings: list[str]
    weights: pd.Series
    entry_prices: pd.Series
    exit_idx: int
    exit_date: pd.Timestamp
    gross: float
    turnover: float
    fee_cost: float
    slippage_cost: float


@dataclass(frozen=True)
class BacktestPeriodResult:
    gross: float
    net: float
    turnover: float
    fee_cost: float
    slippage_cost: float
    total_cost: float
    exit_idx: int
    exit_date: pd.Timestamp


@dataclass(frozen=True)
class BacktestPeriodPlan:
    entry_idx: int
    planned_exit_idx: int
    entry_date: pd.Timestamp
    planned_exit_date: pd.Timestamp
