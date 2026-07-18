"""Validation helpers for optional portfolio selection controls."""

from __future__ import annotations

import math
import operator
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from .execution import SelectionConstraints

MaxNewNamesShortfallPolicy = Literal["legacy_concentrate", "carry", "fail"]
SelectionPricePolicy = Literal["execution_aware", "target_first"]
TargetWeightPolicy = Literal["normalized", "fixed_slot"]


def ranked_selection_frame(
    day: pd.DataFrame,
    pred_col: str,
    *,
    ascending: bool,
    selection_tiebreak_col: str | None = None,
    selection_score_bucket_size: float | None = None,
) -> pd.DataFrame:
    """Return the stable candidate ranking used by selection and weighting."""

    sort_frame = day.copy()
    sort_cols: list[str] = []
    ascending_flags: list[bool] = []
    score_bucket_size = (
        float(selection_score_bucket_size) if selection_score_bucket_size is not None else None
    )
    if score_bucket_size is not None and score_bucket_size <= 0:
        raise ValueError("selection_score_bucket_size must be > 0 when provided.")
    if score_bucket_size is not None:
        score = pd.to_numeric(sort_frame[pred_col], errors="coerce")
        sort_frame["_selection_score_bucket"] = np.floor(score / score_bucket_size)
        sort_cols.append("_selection_score_bucket")
        ascending_flags.append(ascending)
    else:
        sort_cols.append(pred_col)
        ascending_flags.append(ascending)

    if selection_tiebreak_col:
        if selection_tiebreak_col not in sort_frame.columns:
            raise ValueError(f"Selection tiebreaker column not found: {selection_tiebreak_col}")
        sort_frame["_selection_tiebreak"] = pd.to_numeric(
            sort_frame[selection_tiebreak_col],
            errors="coerce",
        ).fillna(-np.inf)
        sort_cols.append("_selection_tiebreak")
        ascending_flags.append(False)
    if score_bucket_size is not None:
        sort_cols.append(pred_col)
        ascending_flags.append(ascending)
    sort_cols.append("symbol")
    ascending_flags.append(True)
    return sort_frame.sort_values(sort_cols, ascending=ascending_flags, kind="mergesort")


def controlled_selection_day(
    day: pd.DataFrame,
    pred_col: str,
    *,
    ascending: bool,
    selection_tiebreak_col: str | None,
    selection_score_bucket_size: float | None,
    selection_min_score: float | None,
    max_new_names_per_rebalance: int | None,
) -> pd.DataFrame:
    """Deduplicate controlled selections using their exact stable ranking order."""

    if selection_min_score is None and max_new_names_per_rebalance is None:
        return day
    ranked = ranked_selection_frame(
        day,
        pred_col,
        ascending=ascending,
        selection_tiebreak_col=selection_tiebreak_col,
        selection_score_bucket_size=selection_score_bucket_size,
    )
    return ranked.drop_duplicates(subset=["symbol"], keep="first")


def merge_pricing_supplemental_columns(
    data: pd.DataFrame,
    pricing_source: pd.DataFrame,
    supplemental_cols: list[str],
) -> pd.DataFrame:
    """Add selection-only pricing columns without changing the core signal frame."""

    if not supplemental_cols:
        return data
    return data.merge(
        pricing_source[["trade_date", "symbol", *supplemental_cols]],
        on=["trade_date", "symbol"],
        how="left",
    )


def apply_liquidity_floor_to_day(
    day: pd.DataFrame,
    *,
    liquidity_floor_col: str | None,
    liquidity_floor_quantile: float | None,
) -> pd.DataFrame:
    """Apply the optional cross-sectional liquidity floor to one signal day."""

    if not liquidity_floor_col or liquidity_floor_quantile is None:
        return day
    if liquidity_floor_col not in day.columns:
        raise ValueError(f"Portfolio liquidity floor column not found: {liquidity_floor_col}")
    floor_q = float(liquidity_floor_quantile)
    if floor_q <= 0:
        return day
    liquidity = pd.to_numeric(day[liquidity_floor_col], errors="coerce")
    if liquidity.notna().sum() <= 1:
        return day
    cutoff = liquidity.quantile(floor_q)
    return day.loc[liquidity.isna() | (liquidity >= cutoff)].copy()


def apply_selection_score_threshold(
    ranked: pd.DataFrame,
    pred_col: str,
    *,
    ascending: bool,
    selection_min_score: float | None,
) -> pd.DataFrame:
    """Keep only candidates on the eligible side of an optional score threshold."""

    if selection_min_score is None:
        return ranked
    scores = pd.to_numeric(ranked[pred_col], errors="coerce")
    eligible = scores <= selection_min_score if ascending else scores >= selection_min_score
    return ranked.loc[eligible].copy()


def entry_amount_values(
    *,
    constraints: SelectionConstraints,
    amount_table: pd.DataFrame | None,
    lookup_date: pd.Timestamp,
) -> pd.Series | None:
    """Resolve the optional entry-date liquidity series."""

    if constraints.min_amount is None:
        return None
    if amount_table is None or lookup_date not in amount_table.index:
        return None
    return amount_table.loc[lookup_date]


def entry_tradable_flags(
    tradable_table: pd.DataFrame | None,
    lookup_date: pd.Timestamp,
) -> pd.Series | None:
    """Resolve the optional entry-date tradability series."""

    if tradable_table is None or lookup_date not in tradable_table.index:
        return None
    return tradable_table.loc[lookup_date]


def validate_selection_min_score(value: Any) -> float | None:
    """Return a finite score threshold or reject ambiguous values."""

    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("selection_min_score must be a finite number when provided.")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("selection_min_score must be a finite number when provided.") from exc
    if not math.isfinite(normalized):
        raise ValueError("selection_min_score must be finite when provided.")
    return normalized


def validate_max_new_names_per_rebalance(value: Any) -> int | None:
    """Return a non-negative integer replacement budget."""

    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("max_new_names_per_rebalance must be a non-negative integer.")
    try:
        normalized = operator.index(value)
    except TypeError as exc:
        raise ValueError("max_new_names_per_rebalance must be a non-negative integer.") from exc
    if normalized < 0:
        raise ValueError("max_new_names_per_rebalance must be >= 0 when provided.")
    return int(normalized)


def validate_max_new_names_shortfall_policy(value: Any) -> MaxNewNamesShortfallPolicy:
    """Return the explicit action when a new-name budget underfills Top-K."""

    normalized = str(value or "legacy_concentrate").strip().lower()
    allowed = {"legacy_concentrate", "carry", "fail"}
    if normalized not in allowed:
        raise ValueError(
            "max_new_names_shortfall_policy must be one of: legacy_concentrate, carry, fail."
        )
    return cast(MaxNewNamesShortfallPolicy, normalized)


def validate_max_positive_names(value: Any) -> int | None:
    """Return a positive integer final-position invariant when configured."""

    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("max_positive_names must be a positive integer.")
    try:
        normalized = operator.index(value)
    except TypeError as exc:
        raise ValueError("max_positive_names must be a positive integer.") from exc
    if normalized <= 0:
        raise ValueError("max_positive_names must be > 0 when provided.")
    return int(normalized)


def validate_entry_rank_cutoff(value: Any) -> int | None:
    """Return an absolute strict entry rank, or reject ambiguous values."""

    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("entry_rank_cutoff must be a positive integer when provided.")
    try:
        normalized = operator.index(value)
    except TypeError as exc:
        raise ValueError("entry_rank_cutoff must be a positive integer when provided.") from exc
    if normalized <= 0:
        raise ValueError("entry_rank_cutoff must be > 0 when provided.")
    return int(normalized)


def validate_selection_price_policy(value: Any) -> SelectionPricePolicy:
    """Normalize whether prices may influence target selection."""

    normalized = str(value or "execution_aware").strip().lower()
    allowed = {"execution_aware", "target_first"}
    if normalized not in allowed:
        raise ValueError("selection_price_policy must be one of: execution_aware, target_first.")
    return cast(SelectionPricePolicy, normalized)


def validate_target_weight_policy(value: Any) -> TargetWeightPolicy:
    """Normalize the target-weight policy without changing the legacy default."""

    normalized = str(value or "normalized").strip().lower()
    allowed = {"normalized", "fixed_slot"}
    if normalized not in allowed:
        raise ValueError("target_weight_policy must be one of: normalized, fixed_slot.")
    return cast(TargetWeightPolicy, normalized)


__all__ = [
    "SelectionPricePolicy",
    "TargetWeightPolicy",
    "apply_liquidity_floor_to_day",
    "apply_selection_score_threshold",
    "controlled_selection_day",
    "entry_amount_values",
    "entry_tradable_flags",
    "merge_pricing_supplemental_columns",
    "ranked_selection_frame",
    "validate_entry_rank_cutoff",
    "validate_max_new_names_per_rebalance",
    "validate_max_new_names_shortfall_policy",
    "validate_max_positive_names",
    "validate_selection_min_score",
    "validate_selection_price_policy",
    "validate_target_weight_policy",
]
