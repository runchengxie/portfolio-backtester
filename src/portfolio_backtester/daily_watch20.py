"""Construct a strict daily 4+16 stock watchlist from precomputed scores.

The module is intentionally model-agnostic.  It consumes one cross-section of
scores and applies portfolio-level constraints; alpha training and score
calculation remain outside ``portfolio-backtester``.

Public symbols are re-exported here; their definitions live in the private
submodules :mod:`_daily_watch20_config` and :mod:`_daily_watch20_select`.
Original behavior is unchanged.
"""

from __future__ import annotations

from ._daily_watch20_config import (
    DailyWatch20Config as DailyWatch20Config,
    DailyWatch20Receipt as DailyWatch20Receipt,
    DailyWatch20Result as DailyWatch20Result,
    DailyWatch20SelectionError as DailyWatch20SelectionError,
    FallbackMode as FallbackMode,
    GuardFactorSpec as GuardFactorSpec,
    ReceiptStatus as ReceiptStatus,
)
from ._daily_watch20_select import select_daily_watch20 as select_daily_watch20

__all__ = [
    "DailyWatch20Config",
    "DailyWatch20Receipt",
    "DailyWatch20Result",
    "DailyWatch20SelectionError",
    "FallbackMode",
    "GuardFactorSpec",
    "ReceiptStatus",
    "select_daily_watch20",
]
