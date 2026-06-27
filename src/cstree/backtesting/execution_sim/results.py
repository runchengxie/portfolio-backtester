from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

__all__ = ["ExecutionAdjustedNavResult", "ExecutionSimResult"]


@dataclass(frozen=True)
class ExecutionSimResult:
    summary: dict[str, Any]
    orders: pd.DataFrame
    fills: pd.DataFrame


@dataclass(frozen=True)
class ExecutionAdjustedNavResult:
    summary: dict[str, Any]
    daily: pd.DataFrame
    orders: pd.DataFrame
    fills: pd.DataFrame
