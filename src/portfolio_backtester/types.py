from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .execution import CostModel, EntryPolicy, ExitPolicy, SelectionConstraints, SlippageModel
from .turnover import TurnoverBreakdown


@dataclass(frozen=True)
class CostBreakdown:
    """Explicit fee and implicit slippage components for a backtest result.

    Stage 3 of the accounting/execution roadmap splits transaction cost into
    eight mutually non-overlapping sub-items. To keep every existing caller
    working without modification, ``fee_cost`` and ``slippage_cost`` remain the
    primary constructor inputs (and ``total_cost`` is always ``fee_cost +
    slippage_cost``). The eight sub-items are optional and default to 0.0, so a
    caller that only knows the aggregate fee/slippage still builds a valid
    breakdown. Use :meth:`from_components` when the eight sub-items are known;
    it derives ``fee_cost``/``slippage_cost`` as aggregates so all three sums
    stay consistent.

    Aggregation rules::

        fee_cost = commission + stamp_tax + transfer_fee
        slippage_cost = spread_cost + temporary_impact + permanent_impact
                        + opportunity_cost + financing_cost
        total_cost = fee_cost + slippage_cost  (== sum of all eight)
    """

    fee_cost: float = 0.0
    slippage_cost: float = 0.0

    # Fee sub-items (sum to fee_cost).
    commission: float = 0.0
    stamp_tax: float = 0.0
    transfer_fee: float = 0.0

    # Slippage / implicit-cost sub-items (sum to slippage_cost).
    spread_cost: float = 0.0
    temporary_impact: float = 0.0
    permanent_impact: float = 0.0
    opportunity_cost: float = 0.0
    financing_cost: float = 0.0

    @classmethod
    def from_components(
        cls,
        *,
        commission: float = 0.0,
        stamp_tax: float = 0.0,
        transfer_fee: float = 0.0,
        spread_cost: float = 0.0,
        temporary_impact: float = 0.0,
        permanent_impact: float = 0.0,
        opportunity_cost: float = 0.0,
        financing_cost: float = 0.0,
    ) -> CostBreakdown:
        """Build from the eight sub-items, deriving the aggregate sums.

        Use this when the underlying engine knows the individual cost
        components. The derived ``fee_cost``/``slippage_cost`` keep
        ``total_cost`` equal to the sum of all eight sub-items.
        """
        fee_cost = float(commission + stamp_tax + transfer_fee)
        slippage_cost = float(
            spread_cost + temporary_impact + permanent_impact + opportunity_cost + financing_cost
        )
        return cls(
            fee_cost=fee_cost,
            slippage_cost=slippage_cost,
            commission=float(commission),
            stamp_tax=float(stamp_tax),
            transfer_fee=float(transfer_fee),
            spread_cost=float(spread_cost),
            temporary_impact=float(temporary_impact),
            permanent_impact=float(permanent_impact),
            opportunity_cost=float(opportunity_cost),
            financing_cost=float(financing_cost),
        )

    @property
    def total_cost(self) -> float:
        return float(self.fee_cost + self.slippage_cost)

    def to_dict(self) -> dict[str, float]:
        return {
            "fee_cost": float(self.fee_cost),
            "slippage_cost": float(self.slippage_cost),
            "commission": float(self.commission),
            "stamp_tax": float(self.stamp_tax),
            "transfer_fee": float(self.transfer_fee),
            "spread_cost": float(self.spread_cost),
            "temporary_impact": float(self.temporary_impact),
            "permanent_impact": float(self.permanent_impact),
            "opportunity_cost": float(self.opportunity_cost),
            "financing_cost": float(self.financing_cost),
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
    target_holdings: set[str] | None = None
    target_weights: pd.Series | None = None


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
    target_name_turnover: float | None = None
    target_entered_names: tuple[str, ...] = ()
    target_exited_names: tuple[str, ...] = ()
    target_overlap_names: tuple[str, ...] = ()
    target_weight_full_l1: float | None = None
    target_weight_half_l1: float | None = None
    pretrade_demand_buy: float | None = None
    pretrade_demand_sell: float | None = None
    pretrade_demand_full_l1: float | None = None
    pretrade_demand_half_l1: float | None = None
    executed_buy: float | None = None
    executed_sell: float | None = None
    executed_gross: float | None = None
    executed_full_l1: float | None = None
    executed_half_l1: float | None = None
    executed_cost: float | None = None
    target_holdings: tuple[str, ...] = ()
    target_weights: pd.Series = field(default_factory=pd.Series)
    target_gross_exposure: float = 1.0
    target_cash_weight: float = 0.0
    modeled_gross_exposure: float = 1.0
    modeled_cash_weight: float = 0.0

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
    target_name_turnover: float | None = None
    target_entered_names: tuple[str, ...] = ()
    target_exited_names: tuple[str, ...] = ()
    target_overlap_names: tuple[str, ...] = ()
    target_weight_full_l1: float | None = None
    target_weight_half_l1: float | None = None
    pretrade_demand_buy: float | None = None
    pretrade_demand_sell: float | None = None
    pretrade_demand_full_l1: float | None = None
    pretrade_demand_half_l1: float | None = None
    executed_buy: float | None = None
    executed_sell: float | None = None
    executed_gross: float | None = None
    executed_full_l1: float | None = None
    executed_half_l1: float | None = None
    executed_cost: float | None = None
    is_initial_build: bool = False
    target_gross_exposure: float | None = None
    target_cash_weight: float | None = None
    modeled_gross_exposure: float | None = None
    modeled_cash_weight: float | None = None

    @property
    def cost_breakdown(self) -> CostBreakdown:
        return CostBreakdown(self.fee_cost, self.slippage_cost)


@dataclass(frozen=True)
class BacktestPeriodPlan:
    entry_idx: int
    planned_exit_idx: int
    entry_date: pd.Timestamp
    planned_exit_date: pd.Timestamp
