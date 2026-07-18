from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .execution import SelectionConstraints
from .holding_selection import select_candidate_holdings
from .selection_controls import (
    MaxNewNamesShortfallPolicy,
    SelectionPricePolicy,
    apply_selection_score_threshold,
    entry_amount_values,
    entry_tradable_flags,
    ranked_selection_frame,
    validate_entry_rank_cutoff,
    validate_max_new_names_per_rebalance,
    validate_max_new_names_shortfall_policy,
    validate_selection_min_score,
    validate_selection_price_policy,
)


def apply_rebalance_buffer(
    ranked_codes: list[str],
    prev_holdings: set[str] | None,
    k: int,
    buffer_exit: int,
    buffer_entry: int,
    entry_rank_cutoff: int | None = None,
) -> list[str]:
    if not ranked_codes or k <= 0:
        return []
    strict_cutoff = validate_entry_rank_cutoff(entry_rank_cutoff)
    if prev_holdings is None:
        if strict_cutoff is not None:
            return list(ranked_codes[:strict_cutoff])
        return list(ranked_codes)
    if buffer_exit <= 0 and buffer_entry <= 0 and strict_cutoff is None:
        return list(ranked_codes)

    keep_limit = min(len(ranked_codes), k + max(0, buffer_exit))
    entry_limit = min(len(ranked_codes), max(0, k - max(0, buffer_entry)))

    keep_set = set(ranked_codes[:keep_limit]) & prev_holdings
    candidate_order: list[str] = [code for code in ranked_codes if code in keep_set]

    preferred_limit = min(strict_cutoff, entry_limit) if strict_cutoff is not None else entry_limit
    preferred = set(ranked_codes[:preferred_limit]) if preferred_limit > 0 else set()
    for code in ranked_codes:
        if len(candidate_order) >= k:
            break
        if code in candidate_order:
            continue
        if strict_cutoff is not None and code not in preferred:
            continue
        if strict_cutoff is None and preferred and code not in preferred:
            continue
        candidate_order.append(code)

    if len(candidate_order) < k and strict_cutoff is None:
        for code in ranked_codes:
            if len(candidate_order) >= k:
                break
            if code not in candidate_order:
                candidate_order.append(code)

    return candidate_order


def apply_rank_offset(ranked_codes: list[str], rank_offset: int = 0) -> list[str]:
    offset = int(rank_offset or 0)
    if offset < 0:
        raise ValueError("rank_offset must be >= 0.")
    if offset <= 0:
        return ranked_codes
    return ranked_codes[offset:]


def _apply_score_margin_holdover(
    *,
    candidate_order: list[str],
    ranked_codes: list[str],
    ranked_scores: dict[str, float],
    prev_holdings: set[str] | None,
    k: int,
    ascending: bool,
    score_margin: float | None,
    score_margin_rank_limit: int | None,
) -> list[str]:
    if not prev_holdings or not score_margin or score_margin <= 0 or k <= 0:
        return candidate_order
    selected = list(dict.fromkeys(candidate_order))[:k]
    selected_set = set(selected)
    rank_limit = int(score_margin_rank_limit or len(ranked_codes))
    rank_map = {code: idx + 1 for idx, code in enumerate(ranked_codes)}
    old_candidates = [
        code
        for code in ranked_codes
        if code in prev_holdings and code not in selected_set and rank_map[code] <= rank_limit
    ]
    for old_code in old_candidates:
        replaceable = [code for code in selected if code not in prev_holdings]
        if not replaceable:
            break
        weakest_new = _weakest_new_symbol(
            replaceable,
            ranked_scores,
            ascending=ascending,
        )
        score_advantage = _score_advantage(
            old_code,
            weakest_new,
            ranked_scores,
            ascending=ascending,
        )
        if score_advantage <= float(score_margin):
            selected[selected.index(weakest_new)] = old_code
            selected_set.remove(weakest_new)
            selected_set.add(old_code)
    selected = sorted(selected, key=lambda code: rank_map.get(code, len(ranked_codes) + 1))
    remainder = [code for code in candidate_order if code not in selected_set]
    if len(remainder) + len(selected) < len(ranked_codes):
        remainder.extend(code for code in ranked_codes if code not in selected_set)
    return selected + list(dict.fromkeys(remainder))


def _weakest_new_symbol(
    replaceable: list[str],
    ranked_scores: dict[str, float],
    *,
    ascending: bool,
) -> str:
    if ascending:
        return max(replaceable, key=lambda code: ranked_scores.get(code, np.inf))
    return min(replaceable, key=lambda code: ranked_scores.get(code, -np.inf))


def _score_advantage(
    old_code: str,
    weakest_new: str,
    ranked_scores: dict[str, float],
    *,
    ascending: bool,
) -> float:
    if ascending:
        return ranked_scores.get(old_code, np.inf) - ranked_scores.get(weakest_new, np.inf)
    return ranked_scores.get(weakest_new, -np.inf) - ranked_scores.get(old_code, -np.inf)


@dataclass(frozen=True)
class _SelectionInputs:
    candidate_order: list[str]
    entry_prices: pd.Series
    amount_values: pd.Series | None
    tradable_flags: pd.Series | None
    group_map: dict[object, object] | None


def select_holdings(
    day: pd.DataFrame,
    entry_date: pd.Timestamp,
    k: int,
    pred_col: str,
    *,
    ascending: bool,
    price_table: pd.DataFrame,
    tradable_table: pd.DataFrame | None,
    amount_table: pd.DataFrame | None,
    constraints: SelectionConstraints | None,
    prev_holdings: set[str] | None,
    buffer_exit: int,
    buffer_entry: int,
    group_col: str | None = None,
    max_names_per_group: int | None = None,
    entry_lookup_date: pd.Timestamp | None = None,
    rank_offset: int = 0,
    selection_tiebreak_col: str | None = None,
    selection_score_bucket_size: float | None = None,
    selection_score_margin: float | None = None,
    selection_score_margin_col: str | None = None,
    selection_score_margin_rank_limit: int | None = None,
    selection_min_score: float | None = None,
    max_new_names_per_rebalance: int | None = None,
    max_new_names_shortfall_policy: MaxNewNamesShortfallPolicy = "legacy_concentrate",
    entry_rank_cutoff: int | None = None,
    selection_price_policy: SelectionPricePolicy = "execution_aware",
) -> tuple[list[str], pd.Series]:
    selection_min_score = validate_selection_min_score(selection_min_score)
    max_new_names_per_rebalance = validate_max_new_names_per_rebalance(max_new_names_per_rebalance)
    max_new_names_shortfall_policy = validate_max_new_names_shortfall_policy(
        max_new_names_shortfall_policy
    )
    entry_rank_cutoff = validate_entry_rank_cutoff(entry_rank_cutoff)
    selection_price_policy = validate_selection_price_policy(selection_price_policy)
    inputs = _build_selection_inputs(
        day=day,
        entry_date=entry_date,
        k=k,
        pred_col=pred_col,
        ascending=ascending,
        price_table=price_table,
        tradable_table=tradable_table,
        amount_table=amount_table,
        constraints=constraints or SelectionConstraints(),
        prev_holdings=prev_holdings,
        buffer_exit=buffer_exit,
        buffer_entry=buffer_entry,
        group_col=group_col,
        max_names_per_group=max_names_per_group,
        entry_lookup_date=entry_lookup_date,
        rank_offset=rank_offset,
        selection_tiebreak_col=selection_tiebreak_col,
        selection_score_bucket_size=selection_score_bucket_size,
        selection_score_margin=selection_score_margin,
        selection_score_margin_col=selection_score_margin_col,
        selection_score_margin_rank_limit=selection_score_margin_rank_limit,
        selection_min_score=selection_min_score,
        entry_rank_cutoff=entry_rank_cutoff,
        selection_price_policy=selection_price_policy,
        deduplicate_symbols=(
            selection_min_score is not None or max_new_names_per_rebalance is not None
        ),
    )
    if inputs is None:
        return [], pd.Series(dtype=float)

    holdings = _select_from_inputs(
        inputs,
        k=k,
        constraints=constraints or SelectionConstraints(),
        max_names_per_group=max_names_per_group,
        prev_holdings=prev_holdings,
        max_new_names_per_rebalance=max_new_names_per_rebalance,
        max_new_names_shortfall_policy=max_new_names_shortfall_policy,
        selection_min_score=selection_min_score,
        selection_price_policy=selection_price_policy,
    )
    if not holdings:
        return [], pd.Series(dtype=float)
    return holdings, inputs.entry_prices.reindex(holdings)


def _select_from_inputs(
    inputs: _SelectionInputs,
    *,
    k: int,
    constraints: SelectionConstraints,
    max_names_per_group: int | None,
    prev_holdings: set[str] | None,
    max_new_names_per_rebalance: int | None,
    max_new_names_shortfall_policy: MaxNewNamesShortfallPolicy,
    selection_min_score: float | None,
    selection_price_policy: SelectionPricePolicy,
) -> list[str]:
    return select_candidate_holdings(
        candidate_order=inputs.candidate_order,
        k=k,
        entry_prices=inputs.entry_prices,
        amount_values=inputs.amount_values,
        tradable_flags=inputs.tradable_flags,
        constraints=constraints or SelectionConstraints(),
        group_map=inputs.group_map,
        max_names_per_group=max_names_per_group,
        prev_holdings=prev_holdings,
        max_new_names_per_rebalance=max_new_names_per_rebalance,
        max_new_names_shortfall_policy=max_new_names_shortfall_policy,
        carry_allowed_symbols=(
            set(inputs.candidate_order) if selection_min_score is not None else None
        ),
        enforce_entry_constraints=selection_price_policy == "execution_aware",
    )


def _build_selection_inputs(
    *,
    day: pd.DataFrame,
    entry_date: pd.Timestamp,
    k: int,
    pred_col: str,
    ascending: bool,
    price_table: pd.DataFrame,
    tradable_table: pd.DataFrame | None,
    amount_table: pd.DataFrame | None,
    constraints: SelectionConstraints,
    prev_holdings: set[str] | None,
    buffer_exit: int,
    buffer_entry: int,
    group_col: str | None,
    max_names_per_group: int | None,
    entry_lookup_date: pd.Timestamp | None,
    rank_offset: int,
    selection_tiebreak_col: str | None,
    selection_score_bucket_size: float | None,
    selection_score_margin: float | None,
    selection_score_margin_col: str | None,
    selection_score_margin_rank_limit: int | None,
    selection_min_score: float | None,
    entry_rank_cutoff: int | None,
    selection_price_policy: SelectionPricePolicy,
    deduplicate_symbols: bool,
) -> _SelectionInputs | None:
    if day.empty or k <= 0:
        return None
    lookup_date = entry_lookup_date or entry_date
    has_entry_row = lookup_date in price_table.index
    if not has_entry_row and selection_price_policy == "execution_aware":
        return None
    entry_prices = price_table.loc[lookup_date] if has_entry_row else pd.Series(dtype=float)

    ranked = ranked_selection_frame(
        day,
        pred_col,
        ascending=ascending,
        selection_tiebreak_col=selection_tiebreak_col,
        selection_score_bucket_size=selection_score_bucket_size,
    )
    if deduplicate_symbols:
        ranked = ranked.drop_duplicates(subset=["symbol"], keep="first")
    ranked = apply_selection_score_threshold(
        ranked,
        pred_col,
        ascending=ascending,
        selection_min_score=selection_min_score,
    )
    ranked_codes = apply_rank_offset(ranked["symbol"].tolist(), rank_offset)
    candidate_order = _candidate_order_with_score_margin(
        ranked=ranked,
        ranked_codes=ranked_codes,
        pred_col=pred_col,
        prev_holdings=prev_holdings,
        k=k,
        ascending=ascending,
        buffer_exit=buffer_exit,
        buffer_entry=buffer_entry,
        selection_score_margin=selection_score_margin,
        selection_score_margin_col=selection_score_margin_col,
        selection_score_margin_rank_limit=selection_score_margin_rank_limit,
        entry_rank_cutoff=entry_rank_cutoff,
    )
    amount_values = entry_amount_values(
        constraints=constraints,
        amount_table=amount_table,
        lookup_date=lookup_date,
    )
    if (
        selection_price_policy == "execution_aware"
        and amount_values is None
        and constraints.min_amount is not None
    ):
        return None
    tradable_flags = entry_tradable_flags(tradable_table, lookup_date)
    if (
        selection_price_policy == "execution_aware"
        and tradable_table is not None
        and tradable_flags is None
    ):
        return None

    return _SelectionInputs(
        candidate_order=candidate_order,
        entry_prices=entry_prices,
        amount_values=amount_values,
        tradable_flags=tradable_flags,
        group_map=_selection_group_map(day, group_col, max_names_per_group),
    )


def _candidate_order_with_score_margin(
    *,
    ranked: pd.DataFrame,
    ranked_codes: list[str],
    pred_col: str,
    prev_holdings: set[str] | None,
    k: int,
    ascending: bool,
    buffer_exit: int,
    buffer_entry: int,
    selection_score_margin: float | None,
    selection_score_margin_col: str | None,
    selection_score_margin_rank_limit: int | None,
    entry_rank_cutoff: int | None,
) -> list[str]:
    candidate_order = apply_rebalance_buffer(
        ranked_codes,
        prev_holdings,
        k,
        buffer_exit,
        buffer_entry,
        entry_rank_cutoff,
    )
    allowed_codes = set(candidate_order)
    margin_col = selection_score_margin_col or pred_col
    if margin_col not in ranked.columns:
        raise ValueError(f"Selection score margin column not found: {margin_col}")
    ranked_scores = dict(
        zip(
            ranked["symbol"],
            pd.to_numeric(ranked[margin_col], errors="coerce"),
            strict=False,
        )
    )
    ordered = _apply_score_margin_holdover(
        candidate_order=candidate_order,
        ranked_codes=ranked_codes,
        ranked_scores=ranked_scores,
        prev_holdings=prev_holdings,
        k=k,
        ascending=ascending,
        score_margin=selection_score_margin,
        score_margin_rank_limit=selection_score_margin_rank_limit,
    )
    return [code for code in ordered if code in allowed_codes]


def _selection_group_map(
    day: pd.DataFrame,
    group_col: str | None,
    max_names_per_group: int | None,
) -> dict[object, object] | None:
    if (
        group_col
        and max_names_per_group is not None
        and max_names_per_group > 0
        and group_col in day.columns
    ):
        return day.set_index("symbol")[group_col].to_dict()
    return None
