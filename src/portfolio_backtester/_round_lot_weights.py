"""Round-lot weighting primitives: variants, numeric coercion, capping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

WeightingMode = Literal["equal", "sqrt_liquidity", "capped_sqrt_liquidity"]


@dataclass(frozen=True)
class RoundLotVariant:
    target_holdings: int
    liquidity_floor_q: float = 0.0
    weighting: WeightingMode = "equal"
    industry_cap: int = 3
    max_weight: float = 0.1
    min_notional: float = 0.0

    @property
    def name(self) -> str:
        return (
            f"h{self.target_holdings}_liq{int(self.liquidity_floor_q * 100):02d}_"
            f"{self.weighting}_icap{self.industry_cap}_max{int(self.max_weight * 100):02d}_"
            f"min{int(self.min_notional / 1000)}k"
        )


def _numeric_series(values: Any, *, index: pd.Index | None = None) -> pd.Series:
    series = pd.Series(values, index=index) if index is not None else pd.Series(values)
    numeric = pd.to_numeric(series, errors="coerce")
    if not isinstance(numeric, pd.Series):
        numeric = pd.Series(numeric, index=series.index)
    return numeric.fillna(0.0)


def _series_float(series: pd.Series, key: str, default: float = np.nan) -> float:
    value = series.get(key, default)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def cap_and_redistribute(raw: pd.Series, cap: float) -> pd.Series:
    """Normalize non-negative weights while capping names when the cap is feasible."""
    base = _numeric_series(raw, index=raw.index).clip(lower=0.0)
    if float(base.sum()) <= 0:
        base = pd.Series(1.0, index=base.index)
    weights = base / float(base.sum())
    if cap <= 0 or cap * len(weights) < 1.0 - 1e-12:
        return weights / float(weights.sum())

    fixed = pd.Series(False, index=weights.index)
    for _ in range(50):
        over_cap = (weights > cap + 1e-12) & (~fixed)
        if not bool(over_cap.any()):
            break
        fixed |= over_cap
        weights.loc[fixed] = cap
        residual = 1.0 - float(weights.loc[fixed].sum())
        free = ~fixed
        if residual <= 1e-12 or not bool(free.any()):
            break
        free_base = base.loc[free]
        if float(free_base.sum()) <= 0:
            weights.loc[free] = residual / int(free.sum())
        else:
            weights.loc[free] = residual * free_base / float(free_base.sum())
    return weights / float(weights.sum())


def _cap_and_redistribute_allow_cash(raw: pd.Series, cap: float) -> pd.Series:
    """Cap weights and leave cash when too few names make the cap infeasible."""
    base = _numeric_series(raw, index=raw.index).clip(lower=0.0)
    if base.empty:
        return base
    if float(base.sum()) <= 0:
        base = pd.Series(1.0, index=base.index)
    if cap <= 0:
        weights = base / float(base.sum())
        return weights / float(weights.sum())

    target_total = min(1.0, float(cap) * len(base))
    weights = target_total * base / float(base.sum())
    fixed = pd.Series(False, index=weights.index)
    for _ in range(50):
        over_cap = (weights > cap + 1e-12) & (~fixed)
        if not bool(over_cap.any()):
            break
        fixed |= over_cap
        weights.loc[fixed] = cap
        residual = target_total - float(weights.loc[fixed].sum())
        free = ~fixed
        if residual <= 1e-12 or not bool(free.any()):
            weights.loc[free] = 0.0
            break
        free_base = base.loc[free]
        if float(free_base.sum()) <= 0:
            weights.loc[free] = residual / int(free.sum())
        else:
            weights.loc[free] = residual * free_base / float(free_base.sum())
    return weights.clip(lower=0.0, upper=cap)
