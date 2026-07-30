"""Order construction submodules (split from orders.py for maintainability)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from ..execution import DetailedTradeFeeModel

TradeFeeModel = DetailedTradeFeeModel


def _target_cash_notional(target_weights: Mapping[str, float], nav: float) -> float:
    if not np.isfinite(float(nav)) or float(nav) <= 0:
        return 0.0
    target_gross = sum(
        max(float(weight), 0.0) for weight in target_weights.values() if np.isfinite(float(weight))
    )
    return max(1.0 - float(target_gross), 0.0) * float(nav)




def _cash_weight_breakdown(
    *,
    cash: float,
    target_cash_notional: float,
    nav: float,
) -> tuple[float, float, float]:
    if not np.isfinite(float(nav)) or float(nav) <= 0:
        return np.nan, np.nan, np.nan
    cash_weight = max(float(cash), 0.0) / float(nav)
    target_cash_weight = min(
        max(float(target_cash_notional), 0.0) / float(nav),
        1.0,
    )
    return (
        float(cash_weight),
        float(target_cash_weight),
        float(max(cash_weight - target_cash_weight, 0.0)),
    )




def _cost_adjusted_target_notional(
    *,
    current_values: Mapping[str, float],
    target_weights: Mapping[str, float],
    nav: float,
    cost_rate: float,
) -> dict[str, float]:
    clean_weights = {
        str(symbol): max(float(weight), 0.0)
        for symbol, weight in target_weights.items()
        if pd.notna(symbol) and np.isfinite(float(weight)) and float(weight) > 0
    }
    if not clean_weights or nav <= 0:
        return {}
    if cost_rate <= 0:
        return {symbol: weight * float(nav) for symbol, weight in clean_weights.items()}

    clean_current = {
        str(symbol): max(float(value), 0.0)
        for symbol, value in current_values.items()
        if pd.notna(symbol) and np.isfinite(float(value)) and float(value) > 0
    }
    symbols = set(clean_current) | set(clean_weights)

    def required_cost(final_nav: float) -> float:
        turnover = 0.0
        for symbol in symbols:
            current_notional = clean_current.get(symbol, 0.0)
            target_notional = clean_weights.get(symbol, 0.0) * final_nav
            turnover += abs(target_notional - current_notional)
        return turnover * float(cost_rate)

    lower = 0.0
    upper = float(nav)
    for _ in range(64):
        mid = (lower + upper) / 2.0
        if mid + required_cost(mid) <= nav:
            lower = mid
        else:
            upper = mid
    return {symbol: weight * lower for symbol, weight in clean_weights.items()}




def _build_targets_by_rebalance(
    positions: pd.DataFrame,
) -> list[tuple[pd.Timestamp, dict[str, Any]]]:
    grouped = []
    for rebalance_date, group in positions.groupby("rebalance_date", sort=True):
        entry_date = pd.to_datetime(group["entry_date"].iloc[0])
        weights = (
            group.groupby("symbol")["weight"]
            .sum()
            .astype(float)
            .loc[lambda series: series > 0]
            .to_dict()
        )
        grouped.append(
            (pd.to_datetime(rebalance_date), {"entry_date": entry_date, "weights": weights})
        )
    return grouped


