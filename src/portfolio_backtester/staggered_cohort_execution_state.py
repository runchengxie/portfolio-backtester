"""State and value primitives for research staggered-cohort execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

TERMINAL_POLICY = "fail_closed"
EPSILON = 1e-8


@dataclass(frozen=True)
class StaggeredCohortExecutionConfig:
    """Frozen research contract for one 1-, 3-, or 5-day execution run."""

    horizon_days: int
    top_n: int
    initial_capital: float = 1_000_000.0
    single_side_cost_bps: float = 10.0
    score_col: str = "score"
    signal_date_col: str = "trade_date"
    available_at_col: str = "available_at"
    valuation_price_col: str = "open"
    terminal_policy: str = TERMINAL_POLICY

    def __post_init__(self) -> None:
        if self.horizon_days not in {1, 3, 5}:
            raise ValueError("staggered execution supports only 1-, 3-, or 5-day horizons")
        if self.top_n <= 0:
            raise ValueError("top_n must be positive")
        if not np.isfinite(self.initial_capital) or self.initial_capital <= 0:
            raise ValueError("initial_capital must be finite and positive")
        valid_cost = (
            np.isfinite(self.single_side_cost_bps)
            and self.single_side_cost_bps >= 0
            and self.single_side_cost_bps < 10_000
        )
        if not valid_cost:
            raise ValueError("single_side_cost_bps must be finite and in [0, 10000)")
        if self.terminal_policy != TERMINAL_POLICY:
            raise ValueError("terminal_policy must be 'fail_closed'")
        valuation_price_col = str(self.valuation_price_col).strip()
        if not valuation_price_col:
            raise ValueError("valuation_price_col must be a non-empty column name")
        object.__setattr__(self, "valuation_price_col", valuation_price_col)


@dataclass(frozen=True)
class StaggeredCohortExecutionResult:
    """Stateful execution outputs; final NAV is absent when liquidation is incomplete."""

    summary: dict[str, Any]
    daily: pd.DataFrame
    positions: pd.DataFrame
    cohort_daily: pd.DataFrame
    orders: pd.DataFrame
    generations: pd.DataFrame


@dataclass
class Position:
    position_id: str
    generation_id: str
    cohort_id: int
    symbol: str
    shares: float
    entry_date: pd.Timestamp
    entry_price: float
    entry_notional: float
    planned_exit_date: pd.Timestamp | None
    planned_exit_idx: int
    last_price: float
    carry_days: int = 0


@dataclass
class Generation:
    generation_id: str
    cohort_id: int
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    planned_exit_date: pd.Timestamp | None
    planned_exit_idx: int
    selected_symbols: tuple[str, ...]
    allocation_budget: float
    entry_notional: float = 0.0
    buy_cost: float = 0.0
    blocked_buy_budget: float = 0.0
    unfunded_buy_budget: float = 0.0
    gross_proceeds: float = 0.0
    sell_cost: float = 0.0
    completed_date: pd.Timestamp | None = None
    status: str = "open"


@dataclass
class Cohort:
    cohort_id: int
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)


@dataclass
class Ledger:
    cohorts: dict[int, Cohort]
    generations: dict[str, Generation] = field(default_factory=dict)
    orders: list[dict[str, Any]] = field(default_factory=list)
    daily: list[dict[str, Any]] = field(default_factory=list)
    positions: list[dict[str, Any]] = field(default_factory=list)
    cohort_daily: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class DayFlow:
    traded_notional: float = 0.0
    transaction_cost: float = 0.0
    blocked_buy_notional: float = 0.0
    blocked_sell_notional: float = 0.0
    suspended_buy_notional: float = 0.0
    suspended_sell_notional: float = 0.0
    unfunded_buy_notional: float = 0.0


def position_value(position: Position) -> float:
    return float(position.shares * position.last_price)


def cohort_nav(cohort: Cohort) -> float:
    return float(cohort.cash + sum(position_value(pos) for pos in cohort.positions.values()))


def ledger_nav(ledger: Ledger) -> float:
    return float(sum(cohort_nav(cohort) for cohort in ledger.cohorts.values()))
