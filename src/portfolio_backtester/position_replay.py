"""Public helpers for canonical position replay inputs and execution."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .backends import (
    CanonicalBacktestResult,
    IntradayExecutionAssumption,
    NativePositionReplayBackend,
    NativePositionReplayRequest,
)
from .execution import SlippageModel
from .execution_sim import ExecutionSimConfig
from .position_backtest import PositionBacktestConfig


def build_position_replay_periods(
    positions: pd.DataFrame,
    pricing: pd.DataFrame,
) -> pd.DataFrame:
    """Build replay periods from position entry dates and available pricing dates."""

    required_positions = {"rebalance_date", "entry_date"}
    missing = sorted(required_positions - set(positions.columns))
    if missing:
        raise ValueError("positions are missing required columns: " + ", ".join(missing))
    if "trade_date" not in pricing.columns:
        raise ValueError("pricing is missing required column: trade_date")
    if positions.empty or pricing.empty:
        return pd.DataFrame(columns=["rebalance_date", "entry_date", "exit_date"])

    work = positions[["rebalance_date", "entry_date"]].copy()
    work["rebalance_date"] = pd.to_datetime(work["rebalance_date"], errors="coerce").dt.normalize()
    work["entry_date"] = pd.to_datetime(work["entry_date"], errors="coerce").dt.normalize()
    if work[["rebalance_date", "entry_date"]].isna().any().any():
        raise ValueError("positions rebalance_date and entry_date must be date-like")

    entries = (
        work.groupby("rebalance_date", as_index=False, sort=True)["entry_date"]
        .min()
        .sort_values("rebalance_date")
        .reset_index(drop=True)
    )
    pricing_dates = pd.to_datetime(pricing["trade_date"], errors="coerce").dt.normalize().dropna()
    if pricing_dates.empty:
        raise ValueError("pricing trade_date must contain at least one valid date")

    next_entries = entries["entry_date"].shift(-1)
    entries["exit_date"] = next_entries.fillna(pricing_dates.max())
    if (entries["exit_date"] < entries["entry_date"]).any():
        raise ValueError("pricing does not cover a valid exit date for every position period")
    return entries[["rebalance_date", "entry_date", "exit_date"]]


def run_native_position_replay(
    positions: pd.DataFrame,
    pricing: pd.DataFrame,
    periods: pd.DataFrame,
    *,
    config: PositionBacktestConfig,
    intraday_bars: pd.DataFrame | None = None,
    intraday_execution_assumption: IntradayExecutionAssumption | None = None,
    allow_stale_execution_price: bool = False,
    ledger: bool = False,
    ledger_config: ExecutionSimConfig | None = None,
    slippage_model: SlippageModel | None = None,
) -> CanonicalBacktestResult:
    """Run standardized positions through the canonical native replay backend."""

    return NativePositionReplayBackend().run(
        NativePositionReplayRequest(
            positions=positions,
            pricing=pricing,
            periods=periods,
            config=config,
            intraday_bars=intraday_bars,
            intraday_execution_assumption=intraday_execution_assumption,
            allow_stale_execution_price=allow_stale_execution_price,
            ledger=ledger,
            ledger_config=ledger_config,
            slippage_model=slippage_model,
        )
    )


__all__ = ["build_position_replay_periods", "run_native_position_replay"]
