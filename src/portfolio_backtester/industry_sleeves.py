"""Portfolio construction helpers for independent industry-balanced sleeves.

The source style-factor research neutralizes signals by industry.  This module
adds the portfolio-layer contract: allocate selection seats within each
industry, restore the eligible universe's industry weights, and combine factor
sleeves only after each sleeve has become a valid long-only portfolio.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

__all__ = [
    "SelectionSpec",
    "attach_entry_dates",
    "build_targets",
    "combine_targets",
    "select_industry_balanced",
    "target_turnover",
    "validate_targets",
]


@dataclass(frozen=True)
class SelectionSpec:
    """Choose either a fixed number of names or a fraction per industry."""

    top_k: int | None = None
    top_fraction: float | None = None

    def __post_init__(self) -> None:
        choices = int(self.top_k is not None) + int(self.top_fraction is not None)
        if choices != 1:
            raise ValueError("exactly one of top_k and top_fraction is required")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.top_fraction is not None and not 0 < self.top_fraction <= 1:
            raise ValueError("top_fraction must be in (0, 1]")


def _seat_counts(counts: pd.Series, top_k: int) -> pd.Series:
    target_k = min(int(top_k), int(counts.sum()))
    raw = counts.astype(float) / float(counts.sum()) * target_k
    seats = np.floor(raw).astype(int).clip(upper=counts)
    minimum = pd.Series(0, index=counts.index, dtype=int)
    if target_k >= len(counts):
        minimum[:] = 1
        seats = seats.clip(lower=minimum)

    while int(seats.sum()) < target_k:
        candidates = counts.index[counts > seats]
        pick = (raw.loc[candidates] - seats.loc[candidates]).idxmax()
        seats.loc[pick] += 1
    while int(seats.sum()) > target_k:
        candidates = counts.index[seats > minimum]
        pick = (raw.loc[candidates] - seats.loc[candidates]).idxmin()
        seats.loc[pick] -= 1
    return seats.astype(int)


def _fractional_seats(counts: pd.Series, fraction: float) -> pd.Series:
    return np.ceil(counts.astype(float) * fraction).astype(int).clip(lower=1, upper=counts)


def select_industry_balanced(
    formation: pd.DataFrame,
    *,
    score_col: str,
    spec: SelectionSpec,
    industry_col: str = "industry_l1",
) -> pd.DataFrame:
    """Select high scores while matching eligible-universe industry weights."""

    required = {"symbol", score_col, industry_col}
    missing = required - set(formation.columns)
    if missing:
        raise ValueError(f"formation is missing columns: {sorted(missing)}")

    work = formation.dropna(subset=["symbol", score_col]).copy()
    work = work.drop_duplicates("symbol", keep="last")
    if work.empty:
        return pd.DataFrame(columns=["symbol", "weight", score_col, industry_col])
    work[industry_col] = work[industry_col].fillna("__UNKNOWN__").astype(str)
    counts = cast(pd.Series, work.groupby(industry_col, sort=True)["symbol"].count())
    if spec.top_k is not None:
        seats = _seat_counts(counts, spec.top_k)
    else:
        seats = _fractional_seats(counts, float(cast(float, spec.top_fraction)))

    selected = []
    for industry, seat_count in seats.items():
        if seat_count == 0:
            continue
        group = work.loc[work[industry_col].eq(industry)]
        chosen = group.nlargest(int(seat_count), score_col, keep="first").copy()
        industry_weight = float(counts.loc[industry] / counts.sum())
        chosen["weight"] = industry_weight / len(chosen)
        selected.append(chosen[["symbol", "weight", score_col, industry_col]])
    result = pd.concat(selected, ignore_index=True)
    result["weight"] /= result["weight"].sum()
    return result.sort_values([industry_col, score_col], ascending=[True, False]).reset_index(
        drop=True
    )


def build_targets(
    panel: pd.DataFrame,
    *,
    score_col: str,
    spec: SelectionSpec,
    date_col: str = "trade_date",
    industry_col: str = "industry_l1",
) -> pd.DataFrame:
    """Build one industry-balanced target portfolio per formation date."""

    required = {date_col, "symbol", score_col, industry_col}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel is missing columns: {sorted(missing)}")
    rows = []
    for date, frame in panel.groupby(date_col, sort=True):
        selected = select_industry_balanced(
            frame,
            score_col=score_col,
            spec=spec,
            industry_col=industry_col,
        )
        if not selected.empty:
            selected.insert(0, "rebalance_date", pd.Timestamp(cast(Any, date)))
            rows.append(selected)
    if not rows:
        return pd.DataFrame(columns=["rebalance_date", "symbol", "weight", score_col, industry_col])
    result = pd.concat(rows, ignore_index=True)
    validate_targets(result)
    return result


def combine_targets(
    sleeves: Mapping[str, pd.DataFrame],
    allocations: Mapping[str, float] | pd.DataFrame,
) -> pd.DataFrame:
    """Mix independently constructed sleeves at the portfolio layer.

    A mapping applies fixed sleeve allocations.  A DataFrame supplies dynamic
    point-in-time allocations with columns ``rebalance_date``, ``sleeve`` and
    ``allocation``.
    """

    if not sleeves:
        raise ValueError("at least one sleeve is required")
    pieces = []
    dynamic = allocations if isinstance(allocations, pd.DataFrame) else None
    for name, targets in sleeves.items():
        part = targets[["rebalance_date", "symbol", "weight"]].copy()
        part["sleeve"] = name
        if dynamic is None:
            if name not in allocations:
                raise ValueError(f"missing allocation for sleeve: {name}")
            part["allocation"] = float(allocations[name])
        else:
            part = part.merge(dynamic, on=["rebalance_date", "sleeve"], how="left")
        if part["allocation"].isna().any():
            raise ValueError(f"missing date allocation for sleeve: {name}")
        part["weight"] *= part["allocation"]
        pieces.append(part)

    combined = pd.concat(pieces, ignore_index=True)
    combined = combined.groupby(["rebalance_date", "symbol"], as_index=False)["weight"].sum()
    totals = combined.groupby("rebalance_date")["weight"].transform("sum")
    combined["weight"] /= totals
    validate_targets(combined)
    return combined.sort_values(["rebalance_date", "weight"], ascending=[True, False])


def attach_entry_dates(targets: pd.DataFrame, trade_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Map each formation date to the next market trading date."""

    calendar = pd.DatetimeIndex(trade_dates).drop_duplicates().sort_values()
    dates = pd.DatetimeIndex(pd.to_datetime(targets["rebalance_date"]))
    indices = calendar.searchsorted(dates, side="right")
    valid = indices < len(calendar)
    result = targets.loc[valid].copy()
    result["entry_date"] = calendar.take(indices[valid]).to_numpy()
    result["side"] = "long"
    return result[["rebalance_date", "entry_date", "symbol", "weight", "side"]]


def target_turnover(targets: pd.DataFrame) -> pd.Series:
    """Half-turnover between successive target portfolios, ignoring drift."""

    raw_dates = sorted(pd.to_datetime(targets["rebalance_date"]).unique())
    dates = [cast(pd.Timestamp, d) for d in raw_dates]
    previous: pd.Series | None = None
    values: dict[pd.Timestamp, float] = {}
    for date in dates:
        current = targets.loc[targets["rebalance_date"].eq(date)].set_index("symbol")["weight"]
        if previous is None:
            values[date] = 1.0
        else:
            aligned = pd.concat([previous, current], axis=1).fillna(0.0)
            absolute_change = (aligned.iloc[:, 1] - aligned.iloc[:, 0]).abs().sum()
            values[date] = float(absolute_change / 2)
        previous = current
    return pd.Series(values, name="target_turnover")


def validate_targets(targets: pd.DataFrame, *, tolerance: float = 1e-8) -> None:
    """Validate the long-only monthly target contract."""

    required = {"rebalance_date", "symbol", "weight"}
    missing = required - set(targets.columns)
    if missing:
        raise ValueError(f"targets are missing columns: {sorted(missing)}")
    if targets.duplicated(["rebalance_date", "symbol"]).any():
        raise ValueError("targets contain duplicate date-symbol rows")
    if (targets["weight"] <= 0).any() or not np.isfinite(targets["weight"]).all():
        raise ValueError("target weights must be finite and positive")
    sums = targets.groupby("rebalance_date")["weight"].sum()
    if not np.allclose(sums.to_numpy(), 1.0, atol=tolerance, rtol=0):
        raise ValueError("target weights must sum to one on every date")
