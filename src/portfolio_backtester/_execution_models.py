"""Execution assumption data models: cost/slippage models and policy dataclasses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
import pandas as pd

from .execution_calendar import MARKET_CALENDAR

ExitPricePolicy = Literal["strict", "ffill", "delay"]
ExitFallbackPolicy = Literal["ffill", "none"]


class CostModel(Protocol):
    def cost(
        self,
        turnover: float,
        *,
        is_initial: bool,
        side: str,
        entry_turnover: float | None = None,
        exit_turnover: float | None = None,
        holding_days: int | None = None,
        gross_exposure: float | None = None,
    ) -> float: ...


class SlippageModel(Protocol):
    def cost(
        self,
        trade_weights: pd.Series,
        *,
        pricing_row: pd.Series | None,
        is_initial: bool,
        side: str,
    ) -> float: ...


@dataclass(frozen=True)
class BpsCostModel:
    bps: float
    round_trip: bool = True

    def cost(
        self,
        turnover: float,
        *,
        is_initial: bool,
        side: str,
        entry_turnover: float | None = None,
        exit_turnover: float | None = None,
        holding_days: int | None = None,
        gross_exposure: float | None = None,
    ) -> float:
        if not np.isfinite(self.bps) or self.bps <= 0:
            return 0.0
        per_side = self.bps / 10000.0
        if is_initial:
            exposure = 1.0 if gross_exposure is None else float(gross_exposure)
            if not np.isfinite(exposure) or exposure < 0:
                exposure = 1.0
            return float(per_side * exposure)
        factor = 2.0 if self.round_trip else 1.0
        return float(factor * per_side * turnover)


@dataclass(frozen=True)
class NoCostModel:
    def cost(
        self,
        turnover: float,
        *,
        is_initial: bool,
        side: str,
        entry_turnover: float | None = None,
        exit_turnover: float | None = None,
        holding_days: int | None = None,
        gross_exposure: float | None = None,
    ) -> float:
        return 0.0


@dataclass(frozen=True)
class SideBpsCostModel:
    long_entry_bps: float
    long_exit_bps: float
    short_entry_bps: float
    short_exit_bps: float
    short_borrow_bps_per_day: float = 0.0

    def cost(
        self,
        turnover: float,
        *,
        is_initial: bool,
        side: str,
        entry_turnover: float | None = None,
        exit_turnover: float | None = None,
        holding_days: int | None = None,
        gross_exposure: float | None = None,
    ) -> float:
        entry = float(entry_turnover) if entry_turnover is not None else float(turnover)
        if exit_turnover is not None:
            exit_ = float(exit_turnover)
        else:
            exit_ = 0.0 if is_initial else float(turnover)

        if side == "short":
            cost = (
                entry * float(self.short_entry_bps) + exit_ * float(self.short_exit_bps)
            ) / 10000.0
            if self.short_borrow_bps_per_day > 0:
                holding = max(0, int(holding_days or 0))
                exposure = float(gross_exposure) if gross_exposure is not None else 1.0
                if np.isfinite(exposure) and exposure > 0 and holding > 0:
                    cost += exposure * holding * float(self.short_borrow_bps_per_day) / 10000.0
            return float(cost)

        return float(
            (entry * float(self.long_entry_bps) + exit_ * float(self.long_exit_bps)) / 10000.0
        )


@dataclass(frozen=True)
class DetailedTradeFeeModel:
    """A-share fee model: commissions, stamp duty, transfer fee, and slippage.

    Default slippage (6 bps buy / 8 bps sell) is calibrated from Level-2
    order-book analysis of ~240 mid-cap A-shares (median half-spread 6.3 bps).
    For price-tiered slippage see ``l2_price_tiered_slippage()``.
    """

    buy_commission_bps: float = 2.5
    sell_commission_bps: float = 2.5
    sell_stamp_duty_bps: float = 5.0
    transfer_fee_bps: float = 0.1
    min_commission: float = 5.0
    buy_slippage_bps: float = 6.0
    sell_slippage_bps: float = 8.0
    portfolio_value: float = 1_000_000.0

    def notional_cost(self, notional: float, *, side: str) -> float:
        amount = max(float(notional), 0.0)
        if amount <= 0:
            return 0.0
        normalized_side = str(side).strip().lower()
        commission_bps = (
            float(self.sell_commission_bps)
            if normalized_side == "sell"
            else float(self.buy_commission_bps)
        )
        commission = amount * max(commission_bps, 0.0) / 10_000.0
        if self.min_commission > 0:
            commission = max(commission, float(self.min_commission))
        slippage_bps = (
            float(self.sell_slippage_bps)
            if normalized_side == "sell"
            else float(self.buy_slippage_bps)
        )
        stamp_bps = float(self.sell_stamp_duty_bps) if normalized_side == "sell" else 0.0
        side_bps = max(slippage_bps, 0.0) + max(stamp_bps, 0.0)
        side_bps += max(float(self.transfer_fee_bps), 0.0)
        return float(commission + amount * side_bps / 10_000.0)

    def cost(
        self,
        turnover: float,
        *,
        is_initial: bool,
        side: str,
        entry_turnover: float | None = None,
        exit_turnover: float | None = None,
        holding_days: int | None = None,
        gross_exposure: float | None = None,
    ) -> float:
        del turnover, is_initial, holding_days
        exposure = float(gross_exposure) if gross_exposure is not None else 1.0
        if not np.isfinite(exposure) or exposure <= 0:
            exposure = 1.0
        portfolio_value = max(float(self.portfolio_value), 1.0)
        entry = max(float(entry_turnover or 0.0), 0.0) * portfolio_value
        exit_ = max(float(exit_turnover or 0.0), 0.0) * portfolio_value
        entry_cost = self.notional_cost(entry, side="buy")
        exit_cost = self.notional_cost(exit_, side="sell")
        return float((entry_cost + exit_cost) / portfolio_value)


@dataclass(frozen=True)
class NoSlippageModel:
    def cost(
        self,
        trade_weights: pd.Series,
        *,
        pricing_row: pd.Series | None,
        is_initial: bool,
        side: str,
    ) -> float:
        return 0.0


@dataclass(frozen=True)
class BpsSlippageModel:
    bps: float

    def cost(
        self,
        trade_weights: pd.Series,
        *,
        pricing_row: pd.Series | None,
        is_initial: bool,
        side: str,
    ) -> float:
        if not np.isfinite(self.bps) or self.bps <= 0:
            return 0.0
        if trade_weights is None or trade_weights.empty:
            return 0.0
        trade_abs = pd.to_numeric(trade_weights, errors="coerce").abs()
        trade_abs = trade_abs[trade_abs.notna()]
        if trade_abs.empty:
            return 0.0
        return float(trade_abs.sum() * float(self.bps) / 10000.0)


@dataclass(frozen=True)
class ParticipationSlippageModel:
    base_bps: float = 0.0
    impact_bps: float = 0.0
    amount_col: str = "amount"
    portfolio_value: float = 1_000_000.0
    power: float = 0.5
    max_participation: float | None = None

    def cost(
        self,
        trade_weights: pd.Series,
        *,
        pricing_row: pd.Series | None,
        is_initial: bool,
        side: str,
    ) -> float:
        if trade_weights is None or trade_weights.empty:
            return 0.0
        trade_abs = pd.to_numeric(trade_weights, errors="coerce").abs()
        trade_abs = trade_abs[trade_abs.notna() & (trade_abs > 0)]
        if trade_abs.empty:
            return 0.0

        per_weight_bps = pd.Series(
            np.repeat(float(self.base_bps), len(trade_abs)),
            index=trade_abs.index,
            dtype=float,
        )
        if (
            pricing_row is not None
            and not pricing_row.empty
            and np.isfinite(self.impact_bps)
            and self.impact_bps > 0
            and np.isfinite(self.portfolio_value)
            and self.portfolio_value > 0
        ):
            amounts = pd.to_numeric(
                pricing_row.reindex(trade_abs.index),
                errors="coerce",
            )
            valid = amounts.notna() & np.isfinite(amounts) & (amounts > 0)
            if valid.any():
                participation = (
                    trade_abs.loc[valid] * float(self.portfolio_value) / amounts.loc[valid]
                )
                participation = participation.clip(lower=0.0)
                if self.max_participation is not None and self.max_participation > 0:
                    participation = participation.clip(upper=float(self.max_participation))
                impact = float(self.impact_bps) * np.power(
                    participation.to_numpy(dtype=float), float(self.power)
                )
                per_weight_bps.loc[valid] = per_weight_bps.loc[valid] + impact
        return float((trade_abs * per_weight_bps / 10000.0).sum())


@dataclass(frozen=True)
class EntryPolicy:
    price_col: str


@dataclass(frozen=True)
class SelectionConstraints:
    min_price: float | None = None
    min_amount: float | None = None
    amount_col: str = "amount"


@dataclass(frozen=True)
class ExitPolicy:
    price_policy: ExitPricePolicy
    fallback_policy: ExitFallbackPolicy
    price_col: str

    def resolve_exit_prices(
        self,
        holdings: list[str],
        planned_exit_idx: int,
        *,
        price_table: pd.DataFrame,
        tradable_table: pd.DataFrame | None,
        trade_dates: list[pd.Timestamp],
        date_to_idx: dict[pd.Timestamp, int],
    ) -> tuple[pd.Series, int]:
        if not holdings:
            return pd.Series(dtype=float), planned_exit_idx

        exit_idx_map: dict[str, int] = {}
        exit_price_map: dict[str, float] = {}
        for symbol in holdings:
            series = price_table[symbol]
            tradable_series = tradable_table[symbol] if tradable_table is not None else None
            exit_idx = self._resolve_exit_idx(
                series,
                planned_exit_idx,
                trade_dates=trade_dates,
                date_to_idx=date_to_idx,
                tradable_series=tradable_series,
            )
            if exit_idx is None:
                continue
            exit_price = price_table.iloc[exit_idx][symbol]
            if not np.isfinite(exit_price):
                continue
            exit_idx_map[symbol] = int(exit_idx)
            exit_price_map[symbol] = float(exit_price)

        if not exit_price_map:
            return pd.Series(dtype=float), planned_exit_idx

        exit_prices = pd.Series(exit_price_map)
        if self.price_policy == "delay":
            max_exit_idx = max(exit_idx_map.values())
            period_exit_idx = max(planned_exit_idx, max_exit_idx)
        else:
            period_exit_idx = planned_exit_idx
        return exit_prices, period_exit_idx

    def _resolve_exit_idx(
        self,
        series: pd.Series,
        planned_exit_idx: int,
        *,
        trade_dates: list[pd.Timestamp],
        date_to_idx: dict[pd.Timestamp, int],
        tradable_series: pd.Series | None,
    ) -> int | None:
        if planned_exit_idx >= len(trade_dates):
            return None
        if self.price_policy == "strict":
            if not np.isfinite(series.iloc[planned_exit_idx]):
                return None
            if tradable_series is not None and not bool(tradable_series.iloc[planned_exit_idx]):
                return None
            return planned_exit_idx

        if self.price_policy == "ffill":
            window = series.iloc[: planned_exit_idx + 1]
            if tradable_series is not None:
                window = window[tradable_series.iloc[: planned_exit_idx + 1]]
            exit_date = window.last_valid_index()
            return date_to_idx.get(exit_date) if exit_date is not None else None

        window = series.iloc[planned_exit_idx:]
        if tradable_series is not None:
            window = window[tradable_series.iloc[planned_exit_idx:]]
        exit_date = window.first_valid_index()
        if exit_date is None and self.fallback_policy == "ffill":
            window = series.iloc[: planned_exit_idx + 1]
            if tradable_series is not None:
                window = window[tradable_series.iloc[: planned_exit_idx + 1]]
            exit_date = window.last_valid_index()
        return date_to_idx.get(exit_date) if exit_date is not None else None


@dataclass(frozen=True)
class ExecutionModel:
    cost_model: CostModel
    slippage_model: SlippageModel
    exit_policy: ExitPolicy
    entry_policy: EntryPolicy
    selection_constraints: SelectionConstraints
    calendar: str = MARKET_CALENDAR
    calendar_open_dates: tuple[pd.Timestamp, ...] = ()
    calendar_closed_dates: tuple[pd.Timestamp, ...] = ()
