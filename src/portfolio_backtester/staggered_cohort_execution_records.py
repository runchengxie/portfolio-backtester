"""Order audit records for staggered-cohort execution."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .staggered_cohort_execution_state import Ledger, Position
from .staggered_cohort_inputs import StaggeredTarget


def raw_audit_fields(
    row: pd.Series | None,
    *,
    valuation_price_col: str,
) -> dict[str, object]:
    if row is None:
        return {
            "raw_open": np.nan,
            "valuation_open": np.nan,
            "valuation_price_col": valuation_price_col,
            "raw_up_limit": np.nan,
            "raw_down_limit": np.nan,
            "raw_is_suspended": pd.NA,
        }
    return {
        "raw_open": row.get("open"),
        "valuation_open": row.get(valuation_price_col),
        "valuation_price_col": valuation_price_col,
        "raw_up_limit": row.get("up_limit"),
        "raw_down_limit": row.get("down_limit"),
        "raw_is_suspended": row.get("is_suspended"),
    }


def append_buy_order(
    ledger: Ledger,
    target: StaggeredTarget,
    symbol: str,
    score: float,
    row: pd.Series | None,
    slot_budget: float,
    requested: float,
    filled: float,
    cost: float,
    reason: str | None,
    *,
    valuation_price_col: str,
) -> None:
    ledger.orders.append(
        {
            "trade_date": target.entry_date,
            "signal_date": target.signal_date,
            "cohort_id": target.cohort_id,
            "generation_id": f"c{target.cohort_id}:{target.entry_date:%Y%m%d}",
            "symbol": symbol,
            "score": score,
            "side": "buy",
            "capital_slot_budget": slot_budget,
            "requested_notional": requested,
            "filled_notional": filled,
            "transaction_cost": cost,
            "status": "filled" if reason is None else "blocked",
            "blocked_reason": reason,
            **raw_audit_fields(row, valuation_price_col=valuation_price_col),
        }
    )


def append_sell_order(
    ledger: Ledger,
    trade_date: pd.Timestamp,
    position: Position,
    row: pd.Series | None,
    requested: float,
    filled: float,
    cost: float,
    reason: str | None,
    *,
    valuation_price_col: str,
) -> None:
    ledger.orders.append(
        {
            "trade_date": trade_date,
            "signal_date": pd.NaT,
            "cohort_id": position.cohort_id,
            "generation_id": position.generation_id,
            "symbol": position.symbol,
            "score": np.nan,
            "side": "sell",
            "capital_slot_budget": np.nan,
            "requested_notional": requested,
            "filled_notional": filled,
            "transaction_cost": cost,
            "status": "filled" if reason is None else "blocked",
            "blocked_reason": reason,
            **raw_audit_fields(row, valuation_price_col=valuation_price_col),
        }
    )
