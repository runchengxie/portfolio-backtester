from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .execution import SelectionConstraints


def apply_rebalance_buffer(
    ranked_codes: list[str],
    prev_holdings: set[str] | None,
    k: int,
    buffer_exit: int,
    buffer_entry: int,
) -> list[str]:
    if not ranked_codes or k <= 0:
        return []
    if prev_holdings is None or (buffer_exit <= 0 and buffer_entry <= 0):
        return list(ranked_codes)

    keep_limit = min(len(ranked_codes), k + max(0, buffer_exit))
    entry_limit = min(len(ranked_codes), max(0, k - max(0, buffer_entry)))

    keep_set = set(ranked_codes[:keep_limit]) & prev_holdings
    candidate_order: list[str] = [code for code in ranked_codes if code in keep_set]

    preferred = set(ranked_codes[:entry_limit]) if entry_limit > 0 else set()
    for code in ranked_codes:
        if len(candidate_order) >= k:
            break
        if code in candidate_order:
            continue
        if preferred and code not in preferred:
            continue
        candidate_order.append(code)

    if len(candidate_order) < k:
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


def _ranked_selection_frame(
    day: pd.DataFrame,
    pred_col: str,
    *,
    ascending: bool,
    selection_tiebreak_col: str | None = None,
    selection_score_bucket_size: float | None = None,
) -> pd.DataFrame:
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

    if score_bucket_size is None:
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


def _passes_entry_constraints(
    symbol: str,
    *,
    entry_prices: pd.Series,
    amount_values: pd.Series | None,
    tradable_flags: pd.Series | None,
    constraints: SelectionConstraints,
) -> bool:
    price = entry_prices.get(symbol, np.nan)
    if not np.isfinite(price):
        return False
    if constraints.min_price is not None and float(price) < float(constraints.min_price):
        return False
    if constraints.min_amount is not None:
        amount = amount_values.get(symbol, np.nan) if amount_values is not None else np.nan
        if not np.isfinite(amount) or float(amount) < float(constraints.min_amount):
            return False
    return not (tradable_flags is not None and not bool(tradable_flags.get(symbol, False)))


def _record_group_selection(
    symbol: str,
    *,
    group_map: dict[object, object],
    group_counts: dict[object, int],
    max_names_per_group: int,
) -> bool:
    group_value = group_map.get(symbol)
    if pd.isna(group_value):
        return True
    current_count = group_counts.get(group_value, 0)
    if current_count >= max_names_per_group:
        return False
    group_counts[group_value] = current_count + 1
    return True


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
    selection_score_margin_rank_limit: int | None = None,
) -> tuple[list[str], pd.Series]:
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
        selection_score_margin_rank_limit=selection_score_margin_rank_limit,
    )
    if inputs is None:
        return [], pd.Series(dtype=float)

    holdings = _select_candidate_holdings(
        candidate_order=inputs.candidate_order,
        k=k,
        entry_prices=inputs.entry_prices,
        amount_values=inputs.amount_values,
        tradable_flags=inputs.tradable_flags,
        constraints=constraints or SelectionConstraints(),
        group_map=inputs.group_map,
        max_names_per_group=max_names_per_group,
    )
    if not holdings:
        return [], pd.Series(dtype=float)
    return holdings, inputs.entry_prices.reindex(holdings)


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
    selection_score_margin_rank_limit: int | None,
) -> _SelectionInputs | None:
    if day.empty or k <= 0:
        return None
    lookup_date = entry_lookup_date or entry_date
    if lookup_date not in price_table.index:
        return None

    ranked = _ranked_selection_frame(
        day,
        pred_col,
        ascending=ascending,
        selection_tiebreak_col=selection_tiebreak_col,
        selection_score_bucket_size=selection_score_bucket_size,
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
        selection_score_margin_rank_limit=selection_score_margin_rank_limit,
    )
    amount_values = _entry_amount_values(
        constraints=constraints,
        amount_table=amount_table,
        lookup_date=lookup_date,
    )
    if amount_values is None and constraints.min_amount is not None:
        return None
    tradable_flags = _entry_tradable_flags(tradable_table, lookup_date)
    if tradable_table is not None and tradable_flags is None:
        return None

    return _SelectionInputs(
        candidate_order=candidate_order,
        entry_prices=price_table.loc[lookup_date],
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
    selection_score_margin_rank_limit: int | None,
) -> list[str]:
    candidate_order = apply_rebalance_buffer(
        ranked_codes,
        prev_holdings,
        k,
        buffer_exit,
        buffer_entry,
    )
    ranked_scores = dict(
        zip(
            ranked["symbol"],
            pd.to_numeric(ranked[pred_col], errors="coerce"),
            strict=False,
        )
    )
    return _apply_score_margin_holdover(
        candidate_order=candidate_order,
        ranked_codes=ranked_codes,
        ranked_scores=ranked_scores,
        prev_holdings=prev_holdings,
        k=k,
        ascending=ascending,
        score_margin=selection_score_margin,
        score_margin_rank_limit=selection_score_margin_rank_limit,
    )


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


def _entry_amount_values(
    *,
    constraints: SelectionConstraints,
    amount_table: pd.DataFrame | None,
    lookup_date: pd.Timestamp,
) -> pd.Series | None:
    if constraints.min_amount is None:
        return None
    if amount_table is None or lookup_date not in amount_table.index:
        return None
    return amount_table.loc[lookup_date]


def _entry_tradable_flags(
    tradable_table: pd.DataFrame | None,
    lookup_date: pd.Timestamp,
) -> pd.Series | None:
    if tradable_table is None:
        return None
    if lookup_date not in tradable_table.index:
        return None
    return tradable_table.loc[lookup_date]


def _select_candidate_holdings(
    *,
    candidate_order: list[str],
    k: int,
    entry_prices: pd.Series,
    amount_values: pd.Series | None,
    tradable_flags: pd.Series | None,
    constraints: SelectionConstraints,
    group_map: dict[object, object] | None,
    max_names_per_group: int | None,
) -> list[str]:
    holdings: list[str] = []
    group_counts: dict[object, int] = {}
    for symbol in candidate_order:
        if len(holdings) >= k:
            break
        if not _passes_entry_constraints(
            symbol,
            entry_prices=entry_prices,
            amount_values=amount_values,
            tradable_flags=tradable_flags,
            constraints=constraints,
        ):
            continue
        if _blocked_by_group_cap(
            symbol,
            group_map=group_map,
            group_counts=group_counts,
            max_names_per_group=max_names_per_group,
        ):
            continue
        holdings.append(symbol)
    return holdings


def _blocked_by_group_cap(
    symbol: str,
    *,
    group_map: dict[object, object] | None,
    group_counts: dict[object, int],
    max_names_per_group: int | None,
) -> bool:
    return bool(
        group_map is not None
        and max_names_per_group is not None
        and not _record_group_selection(
            symbol,
            group_map=group_map,
            group_counts=group_counts,
            max_names_per_group=max_names_per_group,
        )
    )
