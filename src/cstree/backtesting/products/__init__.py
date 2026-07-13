"""Product-specific portfolio selection applications."""

from .daily_watch20 import (
    DailyWatch20Config,
    DailyWatch20Receipt,
    DailyWatch20Result,
    DailyWatch20SelectionError,
    GuardFactorSpec,
    select_daily_watch20,
)

__all__ = [
    "DailyWatch20Config",
    "DailyWatch20Receipt",
    "DailyWatch20Result",
    "DailyWatch20SelectionError",
    "GuardFactorSpec",
    "select_daily_watch20",
]
