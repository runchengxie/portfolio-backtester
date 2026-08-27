"""Shared dataclasses for execution simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..execution import DetailedTradeFeeModel, SlippageModel
from ..types import CostBreakdown
from .config import ExecutionSimConfig

TradeFeeModel = DetailedTradeFeeModel

__all__ = [
    "_AdjustedNavLedger",
    "_AdjustedNavPlan",
    "_ExecutionTables",
    "_MarketRules",
    "_NavOrder",
    "_OrderSink",
    "_trade_fee",
    "describe_trade_fee_model",
]


def describe_trade_fee_model(
    fee_model: TradeFeeModel | None,
    *,
    portfolio_value: float | None = None,
) -> dict[str, Any]:
    if fee_model is None:
        return {"name": "bps"}
    effective_portfolio_value = (
        float(portfolio_value)
        if portfolio_value is not None and np.isfinite(portfolio_value) and portfolio_value > 0
        else float(fee_model.portfolio_value)
    )
    return {
        "name": "detailed",
        "buy_commission_bps": float(fee_model.buy_commission_bps),
        "sell_commission_bps": float(fee_model.sell_commission_bps),
        "sell_stamp_duty_bps": float(fee_model.sell_stamp_duty_bps),
        "transfer_fee_bps": float(fee_model.transfer_fee_bps),
        "min_commission": float(fee_model.min_commission),
        "buy_slippage_bps": float(fee_model.buy_slippage_bps),
        "sell_slippage_bps": float(fee_model.sell_slippage_bps),
        "portfolio_value": effective_portfolio_value,
    }


def _trade_fee(
    notional: float,
    *,
    side: str,
    cost_rate: float,
    fee_model: TradeFeeModel | None,
    slippage_model: SlippageModel | None = None,
    symbol: str | None = None,
    pricing_row: pd.Series | None = None,
    portfolio_value: float | None = None,
) -> CostBreakdown:
    """Per-fill transaction cost split into stage-3 sub-items.

    With a :class:`DetailedTradeFeeModel` the commission/stamp/transfer/spread
    sub-items are taken from ``notional_cost_breakdown``. Without a fee model
    the legacy ``notional * cost_rate`` is treated as an implicit spread cost
    (slippage-like), so ``total_cost`` stays identical to the old scalar path.
    Impact/opportunity/financing sub-items are left at 0 (no model yet).
    """
    if fee_model is None:
        spread = max(float(notional), 0.0) * max(float(cost_rate), 0.0)
        breakdown = CostBreakdown.from_components(spread_cost=spread)
    else:
        fee_breakdown = fee_model.notional_cost_breakdown(notional, side=side)
        breakdown = CostBreakdown.from_components(
            commission=fee_breakdown["commission"],
            stamp_tax=fee_breakdown["stamp_tax"],
            transfer_fee=fee_breakdown["transfer_fee"],
            spread_cost=fee_breakdown["spread_cost"],
        )
    if slippage_model is None or symbol is None or portfolio_value is None or portfolio_value <= 0:
        return breakdown
    trade_weights = pd.Series({symbol: float(notional) / float(portfolio_value)})
    impact = slippage_model.cost(
        trade_weights,
        pricing_row=pricing_row,
        is_initial=False,
        side=side,
    ) * float(portfolio_value)
    if not np.isfinite(impact) or impact <= 0:
        return breakdown
    return CostBreakdown.from_components(
        commission=breakdown.commission,
        stamp_tax=breakdown.stamp_tax,
        transfer_fee=breakdown.transfer_fee,
        spread_cost=breakdown.spread_cost,
        temporary_impact=impact,
        permanent_impact=breakdown.permanent_impact,
        opportunity_cost=breakdown.opportunity_cost,
        financing_cost=breakdown.financing_cost,
    )


def _add_breakdown(total: CostBreakdown, other: CostBreakdown) -> CostBreakdown:
    return CostBreakdown.from_components(
        commission=total.commission + other.commission,
        stamp_tax=total.stamp_tax + other.stamp_tax,
        transfer_fee=total.transfer_fee + other.transfer_fee,
        spread_cost=total.spread_cost + other.spread_cost,
        temporary_impact=total.temporary_impact + other.temporary_impact,
        permanent_impact=total.permanent_impact + other.permanent_impact,
        opportunity_cost=total.opportunity_cost + other.opportunity_cost,
        financing_cost=total.financing_cost + other.financing_cost,
    )


@dataclass(frozen=True)
class _ExecutionTables:
    trade_dates: list[pd.Timestamp]
    date_to_idx: dict[pd.Timestamp, int]
    price_table: pd.DataFrame
    buy_tradable_table: pd.DataFrame | None
    sell_tradable_table: pd.DataFrame | None
    liquidity_tables: dict[str, pd.DataFrame]
    # Phase 4 market-rule tables (None when the corresponding column is absent).
    limit_up_table: pd.DataFrame | None = None
    limit_down_table: pd.DataFrame | None = None
    listing_status_table: pd.DataFrame | None = None


# Public type alias for callers that want to prepare immutable execution tables
# once and reuse them across multiple ledger simulations.
PreparedExecutionTables = _ExecutionTables


@dataclass(frozen=True)
class _MarketRules:
    """Resolved phase-4 market-rule contract for a single simulation run.

    Honors roadmap long-term constraint #7: if a rule is switched on but the
    input it depends on is missing, the run must terminate (``raise``) rather
    than silently downgrade the constraint.
    """

    round_lot: int | None
    enforce_t1: bool
    enforce_price_limits: bool
    enforce_listing_status: bool
    limit_up_table: pd.DataFrame | None
    limit_down_table: pd.DataFrame | None
    listing_status_table: pd.DataFrame | None
    lot_tolerance: float

    def any_active(self) -> bool:
        return (
            self.round_lot is not None
            or self.enforce_t1
            or self.enforce_price_limits
            or self.enforce_listing_status
        )

    @classmethod
    def from_config(
        cls,
        config: ExecutionSimConfig,
        *,
        limit_up_table: pd.DataFrame | None,
        limit_down_table: pd.DataFrame | None,
        listing_status_table: pd.DataFrame | None,
    ) -> _MarketRules:
        if config.enforce_price_limits and (limit_up_table is None or limit_down_table is None):
            raise ValueError(
                "execution_sim.enforce_price_limits requires both limit_up_col and "
                "limit_down_col in the pricing data."
            )
        if config.enforce_listing_status and listing_status_table is None:
            raise ValueError(
                "execution_sim.enforce_listing_status requires listing_status_col in the "
                "pricing data."
            )
        return cls(
            round_lot=config.round_lot,
            enforce_t1=bool(config.enforce_t1),
            enforce_price_limits=bool(config.enforce_price_limits),
            enforce_listing_status=bool(config.enforce_listing_status),
            limit_up_table=limit_up_table,
            limit_down_table=limit_down_table,
            listing_status_table=listing_status_table,
            lot_tolerance=float(config.lot_tolerance),
        )


@dataclass(frozen=True)
class _OrderSink:
    order_rows: list[dict[str, Any]]
    fill_rows: list[dict[str, Any]]


@dataclass
class _NavOrder:
    rebalance_date: pd.Timestamp
    entry_date: pd.Timestamp
    side: str
    symbol: str
    requested_notional: float
    remaining_notional: float
    start_idx: int
    max_days: int
    zero_fill_days: int = 0
    filled_notional: float = 0.0
    first_fill_date: pd.Timestamp | None = None
    last_fill_date: pd.Timestamp | None = None
    fill_days: int = 0
    status: str | None = None
    requested_quantity: float | None = None
    remaining_quantity: float | None = None
    filled_quantity: float = 0.0


@dataclass(frozen=True)
class _AdjustedNavPlan:
    tables: _ExecutionTables
    targets_by_entry: dict[pd.Timestamp, tuple[pd.Timestamp, dict[str, float]]]
    next_entry_by_date: dict[pd.Timestamp, pd.Timestamp | None]
    start_idx: int
    cost_rate: float
    slippage_model: SlippageModel | None = None


@dataclass
class _AdjustedNavLedger:
    cash: float
    previous_nav: float
    target_cash_notional: float
    shares: dict[str, float]
    last_prices: dict[str, float]
    open_orders: list[_NavOrder]
    order_rows: list[dict[str, Any]]
    fill_rows: list[dict[str, Any]]
    daily_rows: list[dict[str, Any]]
    # Phase 4 T+1 ledger: shares available to sell on the current trade date
    # (previous close position; same-day buys are excluded). Refreshed at the
    # start of each trade day before orders execute.
    t1_available: dict[str, float] | None = None
