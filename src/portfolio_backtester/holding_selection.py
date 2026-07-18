from __future__ import annotations

import numpy as np
import pandas as pd

from .execution import SelectionConstraints
from .selection_controls import MaxNewNamesShortfallPolicy


def select_candidate_holdings(
    *,
    candidate_order: list[str],
    k: int,
    entry_prices: pd.Series,
    amount_values: pd.Series | None,
    tradable_flags: pd.Series | None,
    constraints: SelectionConstraints,
    group_map: dict[object, object] | None,
    max_names_per_group: int | None,
    prev_holdings: set[str] | None,
    max_new_names_per_rebalance: int | None,
    max_new_names_shortfall_policy: MaxNewNamesShortfallPolicy,
    carry_allowed_symbols: set[str] | None,
) -> list[str]:
    holdings, group_counts = _select_ranked_holdings(
        candidate_order=candidate_order,
        k=k,
        entry_prices=entry_prices,
        amount_values=amount_values,
        tradable_flags=tradable_flags,
        constraints=constraints,
        group_map=group_map,
        max_names_per_group=max_names_per_group,
        prev_holdings=prev_holdings,
        max_new_names_per_rebalance=max_new_names_per_rebalance,
    )
    _resolve_selection_shortfall(
        holdings,
        prev_holdings=prev_holdings,
        k=k,
        carry_allowed_symbols=carry_allowed_symbols,
        entry_prices=entry_prices,
        amount_values=amount_values,
        tradable_flags=tradable_flags,
        constraints=constraints,
        group_map=group_map,
        group_counts=group_counts,
        max_names_per_group=max_names_per_group,
        max_new_names_per_rebalance=max_new_names_per_rebalance,
        shortfall_policy=max_new_names_shortfall_policy,
    )
    return holdings


def _select_ranked_holdings(
    *,
    candidate_order: list[str],
    k: int,
    entry_prices: pd.Series,
    amount_values: pd.Series | None,
    tradable_flags: pd.Series | None,
    constraints: SelectionConstraints,
    group_map: dict[object, object] | None,
    max_names_per_group: int | None,
    prev_holdings: set[str] | None,
    max_new_names_per_rebalance: int | None,
) -> tuple[list[str], dict[object, int]]:
    holdings: list[str] = []
    group_counts: dict[object, int] = {}
    new_names_selected = 0
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
        is_new_name = prev_holdings is not None and symbol not in prev_holdings
        if _new_name_budget_exhausted(
            is_new_name=is_new_name,
            new_names_selected=new_names_selected,
            max_new_names_per_rebalance=max_new_names_per_rebalance,
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
        if is_new_name:
            new_names_selected += 1
    return holdings, group_counts


def _new_name_budget_exhausted(
    *,
    is_new_name: bool,
    new_names_selected: int,
    max_new_names_per_rebalance: int | None,
) -> bool:
    return bool(
        is_new_name
        and max_new_names_per_rebalance is not None
        and new_names_selected >= max_new_names_per_rebalance
    )


def _resolve_selection_shortfall(
    holdings: list[str],
    *,
    prev_holdings: set[str] | None,
    k: int,
    carry_allowed_symbols: set[str] | None,
    entry_prices: pd.Series,
    amount_values: pd.Series | None,
    tradable_flags: pd.Series | None,
    constraints: SelectionConstraints,
    group_map: dict[object, object] | None,
    group_counts: dict[object, int],
    max_names_per_group: int | None,
    max_new_names_per_rebalance: int | None,
    shortfall_policy: MaxNewNamesShortfallPolicy,
) -> None:
    has_limited_rebalance = prev_holdings is not None and max_new_names_per_rebalance is not None
    if len(holdings) >= k or not has_limited_rebalance:
        return
    if shortfall_policy == "carry":
        _carry_previous_holdings(
            holdings,
            prev_holdings=prev_holdings or set(),
            k=k,
            carry_allowed_symbols=carry_allowed_symbols,
            entry_prices=entry_prices,
            amount_values=amount_values,
            tradable_flags=tradable_flags,
            constraints=constraints,
            group_map=group_map,
            group_counts=group_counts,
            max_names_per_group=max_names_per_group,
        )
        if len(holdings) < k:
            raise ValueError(
                "max_new_names_shortfall_policy='carry' could not restore the target "
                "count from tradable previous holdings."
            )
    if len(holdings) < k and shortfall_policy == "fail":
        raise ValueError(
            "max_new_names_per_rebalance underfilled the target; choose "
            "max_new_names_shortfall_policy='carry' or retain legacy_concentrate."
        )


def _carry_previous_holdings(
    holdings: list[str],
    *,
    prev_holdings: set[str],
    k: int,
    carry_allowed_symbols: set[str] | None,
    entry_prices: pd.Series,
    amount_values: pd.Series | None,
    tradable_flags: pd.Series | None,
    constraints: SelectionConstraints,
    group_map: dict[object, object] | None,
    group_counts: dict[object, int],
    max_names_per_group: int | None,
) -> None:
    for symbol in sorted(prev_holdings):
        if len(holdings) >= k:
            break
        if (
            symbol in holdings
            or (carry_allowed_symbols is not None and symbol not in carry_allowed_symbols)
            or (group_map is not None and symbol not in group_map)
            or not _passes_entry_constraints(
                symbol,
                entry_prices=entry_prices,
                amount_values=amount_values,
                tradable_flags=tradable_flags,
                constraints=constraints,
            )
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
