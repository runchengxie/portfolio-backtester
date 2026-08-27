"""Generic multi-sleeve portfolio construction.

This module owns portfolio mechanics only. Strategy identity and frozen policy
belong to upstream callers, which pass explicit sleeve specifications.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import pandas as pd


@dataclass(frozen=True)
class QuotaSleeveSpec:
    """Select a sleeve by per-group quotas with a rank exit buffer."""

    name: str
    score_col: str
    slots: int
    group_col: str
    quotas: Mapping[str, int]
    exit_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.score_col.strip() or not self.group_col.strip():
            raise ValueError("Sleeve name, score_col, and group_col must be non-empty.")
        if self.slots <= 0:
            raise ValueError("Quota sleeve slots must be positive.")
        if self.exit_multiplier <= 0:
            raise ValueError("Quota sleeve exit_multiplier must be positive.")
        if any(int(value) < 0 for value in self.quotas.values()):
            raise ValueError("Quota sleeve quotas must be non-negative.")


@dataclass(frozen=True)
class RankBufferedSleeveSpec:
    """Select a ranked sleeve with hold/entry ranks and replacement limits."""

    name: str
    score_col: str
    slots: int
    exit_rank: int
    entry_rank: int
    max_replacements: int
    group_col: str | None = None
    max_per_group: int | None = None

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.score_col.strip():
            raise ValueError("Sleeve name and score_col must be non-empty.")
        if self.slots <= 0:
            raise ValueError("Rank-buffered sleeve slots must be positive.")
        if self.exit_rank <= 0 or self.entry_rank <= 0:
            raise ValueError("Rank-buffered sleeve ranks must be positive.")
        if self.max_replacements < 0:
            raise ValueError("max_replacements must be non-negative.")
        if self.max_per_group is not None and self.max_per_group <= 0:
            raise ValueError("max_per_group must be positive when configured.")
        if self.max_per_group is not None and not self.group_col:
            raise ValueError("group_col is required when max_per_group is configured.")


@dataclass(frozen=True)
class SleevePortfolioSpec:
    """Mechanics for combining one quota sleeve and one ranked sleeve."""

    quota_sleeve: QuotaSleeveSpec
    rank_sleeve: RankBufferedSleeveSpec
    overlap_policy: str = "aggregate"
    normal_slot_weight: float = 0.01
    max_name_weight: float = 0.02
    signal_date_col: str = "signal_date"
    trade_date_col: str = "trade_date"
    symbol_col: str = "symbol"

    def __post_init__(self) -> None:
        if self.overlap_policy not in {"aggregate", "deduplicate"}:
            raise ValueError("overlap_policy must be aggregate or deduplicate.")
        if self.normal_slot_weight <= 0 or self.max_name_weight <= 0:
            raise ValueError("Portfolio weights must be positive.")
        if self.max_name_weight < self.normal_slot_weight:
            raise ValueError("max_name_weight must be >= normal_slot_weight.")


def _prepare_signals(signals: pd.DataFrame, spec: SleevePortfolioSpec) -> pd.DataFrame:
    required = {
        spec.symbol_col,
        spec.quota_sleeve.score_col,
        spec.quota_sleeve.group_col,
        spec.rank_sleeve.score_col,
    }
    if spec.rank_sleeve.group_col:
        required.add(spec.rank_sleeve.group_col)
    missing = sorted(required - set(signals.columns))
    if missing:
        raise ValueError("Sleeve signals are missing column(s): " + ", ".join(missing))

    frame = signals.copy()
    if spec.signal_date_col in frame.columns:
        frame[spec.trade_date_col] = pd.to_datetime(
            frame[spec.signal_date_col], errors="coerce"
        )
    elif spec.trade_date_col in frame.columns:
        frame[spec.trade_date_col] = pd.to_datetime(
            frame[spec.trade_date_col], errors="coerce"
        )
    else:
        raise ValueError(
            f"Sleeve signals require {spec.signal_date_col!r} "
            f"or {spec.trade_date_col!r}."
        )
    frame[spec.symbol_col] = frame[spec.symbol_col].astype(str)
    return frame.dropna(subset=[spec.trade_date_col, spec.symbol_col])


def _ranked_symbols(
    frame: pd.DataFrame, *, score_col: str, symbol_col: str
) -> list[str]:
    return frame.sort_values(score_col, ascending=False)[symbol_col].astype(str).tolist()


def _select_quota_sleeve(
    day: pd.DataFrame,
    previous: set[str],
    *,
    spec: QuotaSleeveSpec,
    symbol_col: str,
) -> list[str]:
    selected: list[str] = []
    selected_set: set[str] = set()
    for group_value, raw_quota in spec.quotas.items():
        quota = int(raw_quota)
        if quota <= 0:
            continue
        candidates = day.loc[
            day[spec.group_col].eq(group_value) & day[spec.score_col].notna()
        ]
        ranked = _ranked_symbols(
            candidates, score_col=spec.score_col, symbol_col=symbol_col
        )
        exit_rank = max(quota + 1, int(quota * spec.exit_multiplier))
        pool = [
            symbol
            for rank, symbol in enumerate(ranked, start=1)
            if (symbol in previous and rank <= exit_rank)
            or (symbol not in previous and rank <= quota)
        ][:quota]
        if len(pool) < quota:
            pool.extend(symbol for symbol in ranked if symbol not in pool)
            pool = pool[:quota]
        for symbol in pool:
            if symbol not in selected_set:
                selected.append(symbol)
                selected_set.add(symbol)
    return selected


def _group_value(row: pd.Series, group_col: str | None) -> str:
    if not group_col:
        return ""
    value = row.get(group_col)
    return "" if pd.isna(value) else str(value)


def _select_rank_sleeve(
    day: pd.DataFrame,
    previous: set[str],
    *,
    spec: RankBufferedSleeveSpec,
    symbol_col: str,
) -> list[str]:
    ranked = day.loc[day[spec.score_col].notna()].sort_values(
        spec.score_col, ascending=False
    )
    selected: list[str] = []
    group_counts: dict[str, int] = {}
    replacements = 0

    for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
        symbol = str(row[symbol_col])
        held = symbol in previous
        if held and rank > spec.exit_rank:
            continue
        if not held and (rank > spec.entry_rank or replacements >= spec.max_replacements):
            continue
        group = _group_value(row, spec.group_col)
        if (
            spec.max_per_group is not None
            and group_counts.get(group, 0) >= spec.max_per_group
        ):
            continue
        selected.append(symbol)
        if group:
            group_counts[group] = group_counts.get(group, 0) + 1
        if not held:
            replacements += 1
        if len(selected) >= spec.slots:
            return selected

    # Preserve the historical fill behavior while ownership moves. The fallback
    # intentionally fills by raw rank after the constrained admission pass.
    for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
        symbol = str(row[symbol_col])
        if symbol in selected or rank > spec.exit_rank * 2:
            continue
        selected.append(symbol)
        if len(selected) >= spec.slots:
            break
    return selected


def _fill_sleeve(
    day: pd.DataFrame,
    holdings: list[str],
    *,
    score_col: str,
    slots: int,
    symbol_col: str,
    require_group_col: str | None = None,
) -> list[str]:
    result = list(holdings)
    mask = day[score_col].notna()
    if require_group_col:
        mask &= day[require_group_col].notna()
    candidates = day.loc[mask].sort_values(score_col, ascending=False)
    for symbol in candidates[symbol_col].astype(str):
        if symbol not in result:
            result.append(symbol)
        if len(result) >= slots:
            break
    return result


def _resolve_overlap(
    quota_holdings: list[str],
    rank_holdings: list[str],
    *,
    spec: SleevePortfolioSpec,
) -> tuple[list[str], list[str], dict[str, float]]:
    quota_unique = list(dict.fromkeys(quota_holdings))
    rank_unique = list(dict.fromkeys(rank_holdings))
    overlap = set(quota_unique) & set(rank_unique)
    if spec.overlap_policy == "deduplicate":
        rank_unique = [symbol for symbol in rank_unique if symbol not in overlap]

    weights = dict.fromkeys(quota_unique, spec.normal_slot_weight)
    for symbol in rank_unique:
        weights[symbol] = (
            min(spec.max_name_weight, spec.normal_slot_weight * 2)
            if symbol in overlap and spec.overlap_policy == "aggregate"
            else spec.normal_slot_weight
        )
    return quota_unique, rank_unique, weights


def _build_position_rows(
    date_text: str,
    quota_holdings: list[str],
    rank_holdings: list[str],
    weights: dict[str, float],
    day: pd.DataFrame,
    *,
    spec: SleevePortfolioSpec,
) -> list[dict[str, Any]]:
    lookup = day.drop_duplicates(spec.symbol_col, keep="last").set_index(spec.symbol_col)
    quota_set = set(quota_holdings)
    rank_set = set(rank_holdings)
    carry_columns = [
        column
        for column in dict.fromkeys(
            [
                spec.quota_sleeve.score_col,
                spec.rank_sleeve.score_col,
                spec.quota_sleeve.group_col,
                spec.rank_sleeve.group_col,
            ]
        )
        if column
    ]

    rows: list[dict[str, Any]] = []
    for symbol in sorted(quota_set | rank_set):
        in_quota = symbol in quota_set
        in_rank = symbol in rank_set
        if in_quota and in_rank:
            leg = f"{spec.quota_sleeve.name}+{spec.rank_sleeve.name}"
        elif in_quota:
            leg = spec.quota_sleeve.name
        else:
            leg = spec.rank_sleeve.name
        source = lookup.loc[symbol] if symbol in lookup.index else pd.Series(dtype=object)
        quota_score = pd.to_numeric(
            source.get(spec.quota_sleeve.score_col), errors="coerce"
        )
        rank_score = pd.to_numeric(
            source.get(spec.rank_sleeve.score_col), errors="coerce"
        )
        row: dict[str, Any] = {
            "rebalance_date": date_text,
            "entry_date": date_text,
            "symbol": symbol,
            "weight": weights.get(symbol, 0.0),
            "side": "long",
            "leg": leg,
            "signal": quota_score if pd.notna(quota_score) else rank_score,
        }
        for column in carry_columns:
            row[column] = source.get(column)
        rows.append(row)
    return rows


def build_sleeve_positions(
    signals: pd.DataFrame, *, spec: SleevePortfolioSpec
) -> pd.DataFrame:
    """Convert scored candidates into a positions-by-rebalance frame."""

    frame = _prepare_signals(signals, spec)
    if frame.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    previous_quota: set[str] = set()
    previous_rank: set[str] = set()
    for date in sorted(frame[spec.trade_date_col].unique()):
        day = frame.loc[frame[spec.trade_date_col].eq(date)]
        quota_holdings = _select_quota_sleeve(
            day,
            previous_quota,
            spec=spec.quota_sleeve,
            symbol_col=spec.symbol_col,
        )
        rank_holdings = _select_rank_sleeve(
            day,
            previous_rank,
            spec=spec.rank_sleeve,
            symbol_col=spec.symbol_col,
        )
        quota_holdings = _fill_sleeve(
            day,
            quota_holdings,
            score_col=spec.quota_sleeve.score_col,
            slots=spec.quota_sleeve.slots,
            symbol_col=spec.symbol_col,
            require_group_col=spec.quota_sleeve.group_col,
        )
        rank_holdings = _fill_sleeve(
            day,
            rank_holdings,
            score_col=spec.rank_sleeve.score_col,
            slots=spec.rank_sleeve.slots,
            symbol_col=spec.symbol_col,
        )
        quota_final, rank_final, weights = _resolve_overlap(
            quota_holdings, rank_holdings, spec=spec
        )
        date_text = cast(pd.Timestamp, pd.Timestamp(date)).strftime("%Y%m%d")
        rows.extend(
            _build_position_rows(
                date_text,
                quota_final,
                rank_final,
                weights,
                day,
                spec=spec,
            )
        )
        previous_quota = set(quota_final)
        previous_rank = set(rank_final)

    positions = pd.DataFrame(rows)
    if positions.empty:
        return positions
    positions["rank"] = (
        positions.groupby("rebalance_date", sort=False)["signal"]
        .rank(ascending=False, method="first", na_option="bottom")
        .astype("Int64")
    )
    return positions.sort_values(
        ["rebalance_date", "rank", "symbol"]
    ).reset_index(drop=True)


def compute_position_changes(positions: pd.DataFrame) -> pd.DataFrame:
    """Compare adjacent target snapshots and classify each change."""

    if positions.empty:
        return pd.DataFrame()
    changes: list[dict[str, Any]] = []
    previous: dict[str, float] = {}
    for date in sorted(positions["rebalance_date"].unique()):
        day = positions.loc[positions["rebalance_date"].eq(date)]
        current = dict(zip(day["symbol"], day["weight"], strict=True))
        legs = dict(zip(day["symbol"], day.get("leg", [None] * len(day)), strict=True))
        for symbol in sorted(set(current) | set(previous)):
            weight = float(current.get(symbol, 0.0))
            previous_weight = float(previous.get(symbol, 0.0))
            if weight > 0 and previous_weight == 0:
                action = "new"
            elif weight == 0 and previous_weight > 0:
                action = "exit"
            elif weight != previous_weight:
                action = "weight_change"
            else:
                action = "stay"
            changes.append(
                {
                    "rebalance_date": date,
                    "symbol": symbol,
                    "action": action,
                    "leg": legs.get(symbol),
                    "weight": weight,
                    "prev_weight": previous_weight,
                    "weight_change": weight - previous_weight,
                }
            )
        previous = current
    return pd.DataFrame(changes)


def _leg_mask(legs: pd.Series, token: str) -> pd.Series:
    return legs.fillna("").astype(str).map(lambda value: token in value.split("+"))


def _exposure_summary(positions: pd.DataFrame) -> dict[str, Any]:
    if positions.empty:
        return {}
    legs = positions["leg"].astype("string")
    tokens = sorted(
        {token for value in legs.dropna() for token in str(value).split("+")}
    )
    sleeve_counts = {token: int(_leg_mask(legs, token).sum()) for token in tokens}
    sleeve_weights = {
        token: float(positions.loc[_leg_mask(legs, token), "weight"].sum())
        for token in tokens
    }
    summary: dict[str, Any] = {
        "rebalance_date": str(positions["rebalance_date"].iloc[0]),
        "total_stocks": len(positions),
        "total_weight": float(positions["weight"].sum()),
        "overlap_count": int(legs.fillna("").str.contains("+", regex=False).sum()),
        "sleeve_counts": sleeve_counts,
        "sleeve_weights": sleeve_weights,
    }
    if "A" in tokens:
        summary["a_leg_count"] = sleeve_counts["A"]
        summary["a_weight"] = sleeve_weights["A"]
    if "B" in tokens:
        summary["b_leg_count"] = sleeve_counts["B"]
        summary["b_weight"] = sleeve_weights["B"]
    if "theme" in positions.columns:
        summary["theme_distribution"] = positions["theme"].value_counts().to_dict()
    if "industry" in positions.columns:
        counts = positions["industry"].value_counts().to_dict()
        summary["industry_distribution"] = counts
        if counts:
            largest = max(counts, key=counts.get)
            summary["max_industry_pct"] = round(counts[largest] / len(positions), 4)
    return summary


def compute_position_exposure(positions: pd.DataFrame) -> pd.DataFrame:
    """Compute one portfolio exposure summary for each rebalance date."""

    if positions.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        _exposure_summary(positions.loc[positions["rebalance_date"].eq(date)])
        for date in sorted(positions["rebalance_date"].unique())
    )


__all__ = [
    "QuotaSleeveSpec",
    "RankBufferedSleeveSpec",
    "SleevePortfolioSpec",
    "build_sleeve_positions",
    "compute_position_changes",
    "compute_position_exposure",
]
