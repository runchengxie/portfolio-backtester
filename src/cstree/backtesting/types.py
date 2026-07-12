from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .execution import CostModel, EntryPolicy, ExitPolicy, SelectionConstraints, SlippageModel
from .turnover import TurnoverBreakdown


@dataclass(frozen=True)
class CostBreakdown:
    """Explicit fee and implicit slippage components for a backtest result."""

    fee_cost: float = 0.0
    slippage_cost: float = 0.0

    @property
    def total_cost(self) -> float:
        return float(self.fee_cost + self.slippage_cost)

    def to_dict(self) -> dict[str, float]:
        return {
            "fee_cost": float(self.fee_cost),
            "slippage_cost": float(self.slippage_cost),
            "total_cost": self.total_cost,
        }


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
    buy_turnover: float = 0.0
    sell_turnover: float = 0.0
    gross_traded_weight: float = 0.0
    half_l1_turnover: float = 0.0
    is_initial: bool = False

    @property
    def turnover_breakdown(self) -> TurnoverBreakdown:
        return TurnoverBreakdown(
            buy_weight=self.buy_turnover,
            sell_weight=self.sell_turnover,
            gross_traded_weight=self.gross_traded_weight,
            half_l1_turnover=self.half_l1_turnover,
            one_way_turnover=self.turnover,
            is_initial=self.is_initial,
        )

    @property
    def cost_breakdown(self) -> CostBreakdown:
        return CostBreakdown(self.fee_cost, self.slippage_cost)

    @property
    def total_cost(self) -> float:
        return self.cost_breakdown.total_cost

    @property
    def net(self) -> float:
        return float(self.gross - self.total_cost)


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

    @property
    def cost_breakdown(self) -> CostBreakdown:
        return CostBreakdown(self.fee_cost, self.slippage_cost)


@dataclass(frozen=True)
class BacktestPeriodPlan:
    entry_idx: int
    planned_exit_idx: int
    entry_date: pd.Timestamp
    planned_exit_date: pd.Timestamp
