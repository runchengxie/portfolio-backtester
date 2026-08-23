"""Generic fixed-slot portfolio construction from a pre-ranked candidate frame."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

_REQUIRED_COLUMNS = {
    "symbol",
    "is_current_holding",
    "hold_eligible",
    "entry_eligible",
    "in_entry_pool",
    "overlay_rank",
}


class FixedSlotOverlayError(ValueError):
    """The ranked candidate frame or fixed-slot policy is inconsistent."""


@dataclass(frozen=True, slots=True)
class FixedSlotOverlayTarget:
    """One generic fixed-slot target and its turnover accounting."""

    target_symbols: tuple[str, ...]
    retained_symbols: tuple[str, ...]
    new_symbols: tuple[str, ...]
    exited_symbols: tuple[str, ...]
    target_weights: pd.Series
    target_cash_weight: float
    target_name_turnover: float
    target_full_l1: float
    target_half_l1: float


def _positive_int(value: int, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise FixedSlotOverlayError(f"{field} must be a positive integer")
    return value


def _normalize_symbols(values: tuple[str, ...], *, field: str) -> tuple[str, ...]:
    symbols = tuple(str(value).strip().upper() for value in values)
    if any(not symbol for symbol in symbols):
        raise FixedSlotOverlayError(f"{field} must contain non-empty symbols")
    if len(symbols) != len(set(symbols)):
        raise FixedSlotOverlayError(f"{field} must not contain duplicates")
    return symbols


def _validate_ranking(ranking: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(_REQUIRED_COLUMNS - set(ranking.columns))
    if missing:
        raise FixedSlotOverlayError(f"ranking is missing columns: {missing}")
    if ranking.empty:
        raise FixedSlotOverlayError("ranking must not be empty")
    work = ranking.copy()
    work["symbol"] = work["symbol"].astype("string").str.strip().str.upper()
    if work["symbol"].eq("").any() or work["symbol"].duplicated().any():
        raise FixedSlotOverlayError("ranking symbols must be non-empty and unique")
    for field in ("is_current_holding", "hold_eligible", "entry_eligible", "in_entry_pool"):
        if not work[field].map(lambda value: isinstance(value, bool)).all():
            raise FixedSlotOverlayError(f"ranking {field} must contain booleans")
    ranks = pd.to_numeric(work["overlay_rank"], errors="coerce")
    if ranks.isna().any() or not ranks.gt(0).all() or not ranks.eq(ranks.astype(int)).all():
        raise FixedSlotOverlayError("ranking overlay_rank must contain positive integers")
    work["overlay_rank"] = ranks.astype(int)
    if work["overlay_rank"].duplicated().any():
        raise FixedSlotOverlayError("ranking overlay_rank must be unique")
    return work.sort_values(["overlay_rank", "symbol"], kind="mergesort").reset_index(drop=True)


def build_fixed_slot_overlay_target(
    ranking: pd.DataFrame,
    incumbent_symbols: tuple[str, ...],
    *,
    target_slots: int,
    retain_rank_lte: int,
    new_entry_rank_lte: int,
) -> FixedSlotOverlayTarget:
    """Construct an equal-weight fixed-slot target from an already ranked universe.

    The caller owns signal generation, model ranking, campaign identity and artifact
    validation. This function owns only portfolio selection, weights and turnover.
    """

    slots = _positive_int(target_slots, field="target_slots")
    retain_rank = _positive_int(retain_rank_lte, field="retain_rank_lte")
    entry_rank = _positive_int(new_entry_rank_lte, field="new_entry_rank_lte")
    incumbents = _normalize_symbols(incumbent_symbols, field="incumbent_symbols")
    if len(incumbents) > slots:
        raise FixedSlotOverlayError("incumbent count cannot exceed target_slots")

    work = _validate_ranking(ranking)
    flagged_incumbents = set(work.loc[work["is_current_holding"], "symbol"].astype(str))
    unknown_flagged = flagged_incumbents - set(incumbents)
    if unknown_flagged:
        raise FixedSlotOverlayError(
            "ranking marks symbols as current holdings that are absent from incumbent_symbols"
        )

    retained = list(
        work.loc[
            work["is_current_holding"]
            & work["hold_eligible"]
            & work["overlay_rank"].le(retain_rank),
            "symbol",
        ].astype(str)
    )[:slots]
    additions = list(
        work.loc[
            ~work["is_current_holding"]
            & work["entry_eligible"]
            & work["in_entry_pool"]
            & work["overlay_rank"].le(entry_rank),
            "symbol",
        ].astype(str)
    )
    target = retained.copy()
    for symbol in additions:
        if len(target) >= slots:
            break
        if symbol not in target:
            target.append(symbol)

    target_symbols = tuple(target)
    target_set = set(target_symbols)
    incumbent_set = set(incumbents)
    retained_symbols = tuple(symbol for symbol in target_symbols if symbol in incumbent_set)
    new_symbols = tuple(symbol for symbol in target_symbols if symbol not in incumbent_set)
    exited_symbols = tuple(symbol for symbol in incumbents if symbol not in target_set)

    slot_weight = 1.0 / slots
    weights = pd.Series(slot_weight, index=list(target_symbols), dtype=float)
    previous = pd.Series(slot_weight, index=list(incumbents), dtype=float)
    union = previous.index.union(weights.index)
    full_l1 = float(
        (
            weights.reindex(union).fillna(0.0)
            - previous.reindex(union).fillna(0.0)
        )
        .abs()
        .sum()
    )
    target_cash_weight = 1.0 - float(weights.sum())
    if target_cash_weight < -1e-12:
        raise FixedSlotOverlayError("constructed target weight exceeds one")

    return FixedSlotOverlayTarget(
        target_symbols=target_symbols,
        retained_symbols=retained_symbols,
        new_symbols=new_symbols,
        exited_symbols=exited_symbols,
        target_weights=weights,
        target_cash_weight=target_cash_weight,
        target_name_turnover=(len(new_symbols) + len(exited_symbols)) / (2.0 * slots),
        target_full_l1=full_l1,
        target_half_l1=full_l1 / 2.0,
    )


__all__ = [
    "FixedSlotOverlayError",
    "FixedSlotOverlayTarget",
    "build_fixed_slot_overlay_target",
]
