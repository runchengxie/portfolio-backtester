from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial

import numpy as np
import pandas as pd

from portfolio_backtester._symbol_utils import canonicalize_symbol_columns

from .execution import ExecutionModel, SelectionConstraints
from .execution_calendar import build_execution_date_map
from .portfolio_position_options import PortfolioPositionOptions
from .portfolio_selection import select_holdings
from .portfolio_weights import (
    build_position_weights,
    limit_weight_turnover,
    normalize_position_weights,
    normalize_weighting_mode,
    validate_positive_name_invariant,
)
from .selection_controls import (
    MaxNewNamesShortfallPolicy,
    apply_liquidity_floor_to_day as _apply_liquidity_floor_to_day,
    controlled_selection_day,
    merge_pricing_supplemental_columns as _merge_pricing_supplemental_columns,
    ranked_selection_frame,
    validate_max_new_names_per_rebalance,
    validate_max_new_names_shortfall_policy,
    validate_max_positive_names,
    validate_selection_min_score,
)

POSITION_COLUMNS = [
    "rebalance_date",
    "entry_date",
    "symbol",
    "weight",
    "signal",
    "rank",
    "side",
]


@dataclass(frozen=True)
class PortfolioBuildContext:
    data: pd.DataFrame
    day_groups: dict[pd.Timestamp, pd.DataFrame]
    price_table: pd.DataFrame
    tradable_table: pd.DataFrame | None
    amount_table: pd.DataFrame | None
    trade_dates: list[pd.Timestamp]
    date_to_idx: dict[pd.Timestamp, int]
    explicit_entry_dates: dict[pd.Timestamp, pd.Timestamp]
    calendar_entry_dates: dict[pd.Timestamp, pd.Timestamp]
    selection_constraints: SelectionConstraints


@dataclass
class RebalanceState:
    prev_holdings: set[str] | None = None
    prev_short_holdings: set[str] | None = None
    prev_weights: pd.Series | None = None


@dataclass(frozen=True)
class RebalanceSelection:
    rebalance_date: pd.Timestamp
    entry_date: pd.Timestamp
    entry_lookup_date: pd.Timestamp | None
    day: pd.DataFrame
    k: int


@dataclass(frozen=True)
class PortfolioPositionSetup:
    context: PortfolioBuildContext
    weighting_mode: str


def _empty_positions() -> pd.DataFrame:
    return pd.DataFrame(columns=POSITION_COLUMNS)


def _build_optional_tables(
    pricing_source: pd.DataFrame,
    *,
    tradable_col: str | None,
    selection_constraints: SelectionConstraints,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    tradable_table = None
    if tradable_col and tradable_col in pricing_source.columns:
        tradable_table = pricing_source.pivot(
            index="trade_date", columns="symbol", values=tradable_col
        )
        tradable_table = tradable_table.fillna(False).astype(bool)

    amount_table = None
    amount_col = selection_constraints.amount_col
    if selection_constraints.min_amount is not None:
        if amount_col not in pricing_source.columns:
            raise ValueError(f"Portfolio liquidity column not found: {amount_col}")
        amount_table = pricing_source.pivot(index="trade_date", columns="symbol", values=amount_col)
    return tradable_table, amount_table


def _group_by_trade_date(data: pd.DataFrame) -> dict[pd.Timestamp, pd.DataFrame]:
    groups: dict[pd.Timestamp, pd.DataFrame] = {}
    for date, group in data.groupby("trade_date", sort=False):
        groups[date] = group
    return groups


def _prepare_portfolio_context(
    data: pd.DataFrame,
    *,
    pricing_source: pd.DataFrame,
    entry_price_col: str,
    rebalance_dates: list[pd.Timestamp],
    shift_days: int,
    execution: ExecutionModel | None,
    entry_dates_by_rebalance: dict[pd.Timestamp, pd.Timestamp] | None,
    tradable_col: str | None,
    selection_constraints: SelectionConstraints,
) -> PortfolioBuildContext | None:
    pricing_source = pricing_source.drop_duplicates(subset=["trade_date", "symbol"]).copy()
    if entry_price_col not in pricing_source.columns:
        raise ValueError(f"Portfolio entry price column not found: {entry_price_col}")

    trade_dates = [
        pd.Timestamp(date).normalize() for date in sorted(pricing_source["trade_date"].unique())
    ]
    explicit_entry_dates = {
        pd.Timestamp(key).normalize(): pd.Timestamp(value).normalize()
        for key, value in (entry_dates_by_rebalance or {}).items()
    }
    if len(trade_dates) < 2 and not explicit_entry_dates:
        return None

    date_to_idx = {date: idx for idx, date in enumerate(trade_dates)}
    calendar_entry_dates = {}
    if not explicit_entry_dates and execution is not None:
        calendar_entry_dates = build_execution_date_map(
            rebalance_dates,
            shift_days,
            trade_dates,
            calendar=execution.calendar,
            open_dates=execution.calendar_open_dates,
            closed_dates=execution.calendar_closed_dates,
        )

    tradable_table, amount_table = _build_optional_tables(
        pricing_source,
        tradable_col=tradable_col,
        selection_constraints=selection_constraints,
    )
    return PortfolioBuildContext(
        data=data,
        day_groups=_group_by_trade_date(data),
        price_table=pricing_source.pivot(
            index="trade_date", columns="symbol", values=entry_price_col
        ),
        tradable_table=tradable_table,
        amount_table=amount_table,
        trade_dates=trade_dates,
        date_to_idx=date_to_idx,
        explicit_entry_dates=explicit_entry_dates,
        calendar_entry_dates=calendar_entry_dates,
        selection_constraints=selection_constraints,
    )


def _resolve_rebalance_selection(
    context: PortfolioBuildContext,
    rebalance_date: pd.Timestamp,
    *,
    shift_days: int,
    top_k: int,
    liquidity_floor_col: str | None,
    liquidity_floor_quantile: float | None,
) -> RebalanceSelection | None:
    reb_date = pd.Timestamp(rebalance_date).normalize()
    if reb_date not in context.date_to_idx:
        return None

    entry_date = context.explicit_entry_dates.get(reb_date) or context.calendar_entry_dates.get(
        reb_date
    )
    entry_lookup_date = None
    if entry_date is None:
        entry_idx = context.date_to_idx[reb_date] + shift_days
        if entry_idx >= len(context.trade_dates):
            return None
        entry_date = context.trade_dates[entry_idx]
    entry_date = pd.Timestamp(entry_date).normalize()
    if entry_date not in context.date_to_idx:
        entry_lookup_date = reb_date

    day = context.day_groups.get(reb_date)
    if day is None or day.empty:
        return None
    day = _apply_liquidity_floor_to_day(
        day,
        liquidity_floor_col=liquidity_floor_col,
        liquidity_floor_quantile=liquidity_floor_quantile,
    )
    if day.empty:
        return None

    k = min(int(top_k), len(day))
    if k <= 0:
        return None
    return RebalanceSelection(
        rebalance_date=reb_date,
        entry_date=entry_date,
        entry_lookup_date=entry_lookup_date,
        day=day,
        k=k,
    )


def _rank_and_signal_maps(
    day: pd.DataFrame,
    pred_col: str,
    *,
    ascending: bool,
    selection_tiebreak_col: str | None = None,
    selection_score_bucket_size: float | None = None,
) -> tuple[dict[str, int], dict[object, object]]:
    ranked_codes = ranked_selection_frame(
        day,
        pred_col,
        ascending=ascending,
        selection_tiebreak_col=selection_tiebreak_col,
        selection_score_bucket_size=selection_score_bucket_size,
    )["symbol"].tolist()
    return {code: idx + 1 for idx, code in enumerate(ranked_codes)}, day.set_index("symbol")[
        pred_col
    ].to_dict()


def _append_position_rows(
    results: list[dict[str, object]],
    *,
    selection: RebalanceSelection,
    holdings: list[str],
    weights: pd.Series,
    rank_map: dict[str, int],
    signal_map: dict[object, object],
    side: str,
    weight_sign: float = 1.0,
) -> None:
    for code in holdings:
        results.append(
            {
                "rebalance_date": selection.rebalance_date.strftime("%Y%m%d"),
                "entry_date": selection.entry_date.strftime("%Y%m%d"),
                "symbol": code,
                "weight": float(weight_sign * weights.get(code, 0.0)),
                "signal": float(signal_map.get(code, np.nan)),
                "rank": int(rank_map.get(code, 0)),
                "side": side,
            }
        )


def _select_side_holdings(
    context: PortfolioBuildContext,
    selection: RebalanceSelection,
    pred_col: str,
    *,
    k: int,
    ascending: bool,
    prev_holdings: set[str] | None,
    buffer_exit: int,
    buffer_entry: int,
    group_col: str | None,
    max_names_per_group: int | None,
    rank_offset: int,
    selection_tiebreak_col: str | None,
    selection_score_bucket_size: float | None,
    selection_score_margin: float | None,
    selection_score_margin_col: str | None,
    selection_score_margin_rank_limit: int | None,
    selection_min_score: float | None,
    max_new_names_per_rebalance: int | None,
    max_new_names_shortfall_policy: MaxNewNamesShortfallPolicy,
) -> list[str]:
    holdings, _ = select_holdings(
        selection.day,
        selection.entry_date,
        k,
        pred_col,
        ascending=ascending,
        price_table=context.price_table,
        tradable_table=context.tradable_table,
        amount_table=context.amount_table,
        constraints=context.selection_constraints,
        prev_holdings=prev_holdings,
        buffer_exit=buffer_exit,
        buffer_entry=buffer_entry,
        group_col=group_col,
        max_names_per_group=max_names_per_group,
        entry_lookup_date=selection.entry_lookup_date,
        rank_offset=rank_offset,
        selection_tiebreak_col=selection_tiebreak_col,
        selection_score_bucket_size=selection_score_bucket_size,
        selection_score_margin=selection_score_margin,
        selection_score_margin_col=selection_score_margin_col,
        selection_score_margin_rank_limit=selection_score_margin_rank_limit,
        selection_min_score=selection_min_score,
        max_new_names_per_rebalance=max_new_names_per_rebalance,
        max_new_names_shortfall_policy=max_new_names_shortfall_policy,
    )
    return holdings


def _build_and_append_side(
    results: list[dict[str, object]],
    selection: RebalanceSelection,
    holdings: list[str],
    pred_col: str,
    *,
    side: str,
    weighting_mode: str,
    weighting_liquidity_col: str,
    rank_ascending: bool,
    selection_tiebreak_col: str | None,
    selection_score_bucket_size: float | None,
    selection_min_score: float | None,
    max_new_names_per_rebalance: int | None,
    weight_sign: float = 1.0,
) -> bool:
    selection = replace(
        selection,
        day=controlled_selection_day(
            selection.day,
            pred_col,
            ascending=rank_ascending,
            selection_tiebreak_col=selection_tiebreak_col,
            selection_score_bucket_size=selection_score_bucket_size,
            selection_min_score=selection_min_score,
            max_new_names_per_rebalance=max_new_names_per_rebalance,
        ),
    )
    weights = build_position_weights(
        selection.day,
        holdings,
        pred_col,
        side=side,
        weighting=weighting_mode,
        liquidity_col=weighting_liquidity_col,
    )
    if weights.empty:
        return False
    rank_map, signal_map = _rank_and_signal_maps(
        selection.day,
        pred_col,
        ascending=rank_ascending,
        selection_tiebreak_col=selection_tiebreak_col,
        selection_score_bucket_size=selection_score_bucket_size,
    )
    _append_position_rows(
        results,
        selection=selection,
        holdings=holdings,
        weights=weights,
        rank_map=rank_map,
        signal_map=signal_map,
        side=side,
        weight_sign=weight_sign,
    )
    return True


def _process_long_only_rebalance(
    results: list[dict[str, object]],
    context: PortfolioBuildContext,
    selection: RebalanceSelection,
    state: RebalanceState,
    options: PortfolioPositionOptions,
) -> None:
    holdings = _select_side_holdings(
        context,
        selection,
        options.pred_col,
        k=selection.k,
        ascending=False,
        prev_holdings=state.prev_holdings,
        buffer_exit=options.buffer_exit,
        buffer_entry=options.buffer_entry,
        group_col=options.group_col,
        max_names_per_group=options.max_names_per_group,
        rank_offset=options.rank_offset,
        selection_tiebreak_col=options.selection_tiebreak_col,
        selection_score_bucket_size=options.selection_score_bucket_size,
        selection_score_margin=options.selection_score_margin,
        selection_score_margin_col=options.selection_score_margin_col,
        selection_score_margin_rank_limit=options.selection_score_margin_rank_limit,
        selection_min_score=options.selection_min_score,
        max_new_names_per_rebalance=options.max_new_names_per_rebalance,
        max_new_names_shortfall_policy=options.max_new_names_shortfall_policy,
    )
    if not holdings:
        if (
            options.selection_min_score is not None
            or options.max_new_names_per_rebalance is not None
        ) and state.prev_holdings is not None:
            state.prev_weights = pd.Series(dtype=float)
            state.prev_holdings = set()
        return
    selection = replace(selection, day=options.controlled_day(selection.day, ascending=False))
    weights = build_position_weights(
        selection.day,
        holdings,
        options.pred_col,
        side="long",
        weighting=options.weighting_mode,
        liquidity_col=options.weighting_liquidity_col,
    )
    weights = limit_weight_turnover(
        state.prev_weights,
        weights,
        options.max_turnover_per_rebalance,
    )
    weights = validate_positive_name_invariant(weights, options.max_positive_names)
    if options.selection_min_score is not None:
        weights = normalize_position_weights(weights.reindex(holdings).dropna())
    if weights.empty:
        return
    rank_map, signal_map = _rank_and_signal_maps(
        selection.day,
        options.pred_col,
        ascending=False,
        selection_tiebreak_col=options.selection_tiebreak_col,
        selection_score_bucket_size=options.selection_score_bucket_size,
    )
    _append_position_rows(
        results,
        selection=selection,
        holdings=list(weights.index),
        weights=weights,
        rank_map=rank_map,
        signal_map=signal_map,
        side="long",
    )
    state.prev_weights = weights
    state.prev_holdings = set(weights.index)


def _commit_long_short_state(
    state: RebalanceState,
    *,
    long_holdings: list[str],
    short_holdings: list[str],
    completed: bool,
) -> None:
    if not completed:
        return
    if long_holdings or state.prev_holdings is not None:
        state.prev_holdings = set(long_holdings)
    if short_holdings or state.prev_short_holdings is not None:
        state.prev_short_holdings = set(short_holdings)


def _append_long_short_positions(
    results: list[dict[str, object]],
    selection: RebalanceSelection,
    options: PortfolioPositionOptions,
    *,
    long_holdings: list[str],
    short_holdings: list[str],
) -> bool:
    append_side = partial(
        _build_and_append_side,
        results,
        selection,
        pred_col=options.pred_col,
        weighting_mode=options.weighting_mode,
        weighting_liquidity_col=options.weighting_liquidity_col,
        selection_tiebreak_col=options.selection_tiebreak_col,
        selection_score_bucket_size=options.selection_score_bucket_size,
        selection_min_score=options.selection_min_score,
        max_new_names_per_rebalance=options.max_new_names_per_rebalance,
    )
    long_ok = not long_holdings or append_side(
        long_holdings,
        side="long",
        rank_ascending=False,
    )
    short_ok = not short_holdings or append_side(
        short_holdings,
        side="short",
        rank_ascending=True,
        weight_sign=-1.0,
    )
    return long_ok and short_ok


def _process_long_short_rebalance(
    results: list[dict[str, object]],
    context: PortfolioBuildContext,
    selection: RebalanceSelection,
    state: RebalanceState,
    options: PortfolioPositionOptions,
) -> None:
    controlled = (
        options.selection_min_score is not None or options.max_new_names_per_rebalance is not None
    )
    short_k_final = int(options.short_k if options.short_k is not None else selection.k)
    if not controlled:
        short_k_final = min(short_k_final, len(selection.day) - selection.k)
    if short_k_final <= 0:
        return
    long_holdings = _select_side_holdings(
        context,
        selection,
        options.pred_col,
        k=selection.k,
        ascending=False,
        prev_holdings=state.prev_holdings,
        buffer_exit=options.buffer_exit,
        buffer_entry=options.buffer_entry,
        group_col=options.group_col,
        max_names_per_group=options.max_names_per_group,
        rank_offset=0,
        selection_tiebreak_col=options.selection_tiebreak_col,
        selection_score_bucket_size=options.selection_score_bucket_size,
        selection_score_margin=options.selection_score_margin,
        selection_score_margin_col=options.selection_score_margin_col,
        selection_score_margin_rank_limit=options.selection_score_margin_rank_limit,
        selection_min_score=options.selection_min_score,
        max_new_names_per_rebalance=options.max_new_names_per_rebalance,
        max_new_names_shortfall_policy=options.max_new_names_shortfall_policy,
    )
    short_selection = selection
    if controlled:
        short_day = selection.day.loc[~selection.day["symbol"].isin(long_holdings)].copy()
        short_k_final = min(short_k_final, len(short_day))
        short_selection = replace(selection, day=short_day, k=short_k_final)
    short_holdings = _select_side_holdings(
        context,
        short_selection,
        options.pred_col,
        k=short_k_final,
        ascending=True,
        prev_holdings=state.prev_short_holdings,
        buffer_exit=options.buffer_exit,
        buffer_entry=options.buffer_entry,
        group_col=options.group_col,
        max_names_per_group=options.max_names_per_group,
        rank_offset=0,
        selection_tiebreak_col=options.selection_tiebreak_col,
        selection_score_bucket_size=options.selection_score_bucket_size,
        selection_score_margin=options.selection_score_margin,
        selection_score_margin_col=options.selection_score_margin_col,
        selection_score_margin_rank_limit=options.selection_score_margin_rank_limit,
        selection_min_score=options.selection_min_score,
        max_new_names_per_rebalance=options.max_new_names_per_rebalance,
        max_new_names_shortfall_policy=options.max_new_names_shortfall_policy,
    )
    if not controlled and (not long_holdings or not short_holdings):
        return
    completed = _append_long_short_positions(
        results,
        selection,
        options,
        long_holdings=long_holdings,
        short_holdings=short_holdings,
    )
    _commit_long_short_state(
        state,
        long_holdings=long_holdings,
        short_holdings=short_holdings,
        completed=completed,
    )


def _normalize_portfolio_frames(
    data: pd.DataFrame,
    pricing_data: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    if data is not None and not data.empty:
        data = canonicalize_symbol_columns(data, context="Portfolio data")
        data = data.copy()
        data["trade_date"] = pd.to_datetime(data["trade_date"]).dt.normalize()
    if pricing_data is not None and not pricing_data.empty:
        pricing_data = canonicalize_symbol_columns(
            pricing_data,
            context="Portfolio pricing data",
        )
        pricing_data = pricing_data.copy()
        pricing_data["trade_date"] = pd.to_datetime(pricing_data["trade_date"]).dt.normalize()
    return data, pricing_data


def _resolve_pricing_source(
    data: pd.DataFrame,
    pricing_data: pd.DataFrame | None,
) -> pd.DataFrame | None:
    if pricing_data is not None and not pricing_data.empty:
        return pricing_data
    return data


def _prepare_position_setup(
    data: pd.DataFrame,
    *,
    price_col: str,
    rebalance_dates: list[pd.Timestamp],
    shift_days: int,
    weighting: str,
    execution: ExecutionModel | None,
    entry_dates_by_rebalance: dict[pd.Timestamp, pd.Timestamp] | None,
    pricing_data: pd.DataFrame | None,
    tradable_col: str | None,
    liquidity_floor_col: str | None,
    weighting_liquidity_col: str,
) -> PortfolioPositionSetup | None:
    weighting_mode = normalize_weighting_mode(weighting)
    entry_price_col = execution.entry_policy.price_col if execution is not None else price_col
    selection_constraints = (
        execution.selection_constraints if execution is not None else SelectionConstraints()
    )
    pricing_source = _resolve_pricing_source(data, pricing_data)
    if pricing_source is None or pricing_source.empty:
        return None

    supplemental_cols = [
        col
        for col in {liquidity_floor_col, weighting_liquidity_col}
        if col and col not in data.columns and col in pricing_source.columns
    ]
    data = _merge_pricing_supplemental_columns(data, pricing_source, supplemental_cols)
    context = _prepare_portfolio_context(
        data,
        pricing_source=pricing_source,
        entry_price_col=entry_price_col,
        rebalance_dates=rebalance_dates,
        shift_days=shift_days,
        execution=execution,
        entry_dates_by_rebalance=entry_dates_by_rebalance,
        tradable_col=tradable_col,
        selection_constraints=selection_constraints,
    )
    if context is None:
        return None
    return PortfolioPositionSetup(
        context=context,
        weighting_mode=weighting_mode,
    )


def _build_position_rows_by_rebalance(
    context: PortfolioBuildContext,
    options: PortfolioPositionOptions,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    state = RebalanceState()
    for rebalance_date in options.rebalance_dates:
        selection = _resolve_rebalance_selection(
            context,
            rebalance_date,
            shift_days=options.shift_days,
            top_k=options.top_k,
            liquidity_floor_col=options.liquidity_floor_col,
            liquidity_floor_quantile=options.liquidity_floor_quantile,
        )
        if selection is None:
            continue
        if options.long_only:
            _process_long_only_rebalance(
                results,
                context,
                selection,
                state,
                options,
            )
        else:
            _process_long_short_rebalance(
                results,
                context,
                selection,
                state,
                options,
            )
    return results


def _positions_frame_from_rows(results: list[dict[str, object]]) -> pd.DataFrame:
    if not results:
        return _empty_positions()
    output = pd.DataFrame(results)
    output.sort_values(["entry_date", "side", "rank", "symbol"], inplace=True)
    return output.reset_index(drop=True)


def build_positions_by_rebalance(
    data: pd.DataFrame,
    pred_col: str,
    price_col: str,
    rebalance_dates: list[pd.Timestamp],
    top_k: int,
    shift_days: int,
    *,
    weighting: str = "equal",
    buffer_exit: int = 0,
    buffer_entry: int = 0,
    long_only: bool = True,
    short_k: int | None = None,
    tradable_col: str | None = None,
    group_col: str | None = None,
    max_names_per_group: int | None = None,
    execution: ExecutionModel | None = None,
    entry_dates_by_rebalance: dict[pd.Timestamp, pd.Timestamp] | None = None,
    pricing_data: pd.DataFrame | None = None,
    liquidity_floor_col: str | None = None,
    liquidity_floor_quantile: float | None = None,
    weighting_liquidity_col: str = "medadv20_amount",
    max_turnover_per_rebalance: float | None = None,
    rank_offset: int = 0,
    selection_tiebreak_col: str | None = None,
    selection_score_bucket_size: float | None = None,
    selection_score_margin: float | None = None,
    selection_score_margin_col: str | None = None,
    selection_score_margin_rank_limit: int | None = None,
    selection_min_score: float | None = None,
    max_new_names_per_rebalance: int | None = None,
    max_new_names_shortfall_policy: MaxNewNamesShortfallPolicy = "legacy_concentrate",
    max_positive_names: int | None = None,
) -> pd.DataFrame:
    selection_min_score = validate_selection_min_score(selection_min_score)
    max_new_names_per_rebalance = validate_max_new_names_per_rebalance(max_new_names_per_rebalance)
    max_new_names_shortfall_policy = validate_max_new_names_shortfall_policy(
        max_new_names_shortfall_policy
    )
    max_positive_names = validate_max_positive_names(max_positive_names)
    data, pricing_data = _normalize_portfolio_frames(data, pricing_data)
    if data.empty or not rebalance_dates or top_k <= 0:
        return _empty_positions()
    setup = _prepare_position_setup(
        data,
        price_col=price_col,
        rebalance_dates=rebalance_dates,
        shift_days=shift_days,
        weighting=weighting,
        execution=execution,
        entry_dates_by_rebalance=entry_dates_by_rebalance,
        pricing_data=pricing_data,
        tradable_col=tradable_col,
        liquidity_floor_col=liquidity_floor_col,
        weighting_liquidity_col=weighting_liquidity_col,
    )
    if setup is None:
        return _empty_positions()
    options = PortfolioPositionOptions(
        pred_col=pred_col,
        rebalance_dates=rebalance_dates,
        shift_days=shift_days,
        top_k=top_k,
        weighting_mode=setup.weighting_mode,
        weighting_liquidity_col=weighting_liquidity_col,
        buffer_exit=buffer_exit,
        buffer_entry=buffer_entry,
        long_only=long_only,
        short_k=short_k,
        group_col=group_col,
        max_names_per_group=max_names_per_group,
        liquidity_floor_col=liquidity_floor_col,
        liquidity_floor_quantile=liquidity_floor_quantile,
        max_turnover_per_rebalance=max_turnover_per_rebalance,
        rank_offset=rank_offset,
        selection_tiebreak_col=selection_tiebreak_col,
        selection_score_bucket_size=selection_score_bucket_size,
        selection_score_margin=selection_score_margin,
        selection_score_margin_col=selection_score_margin_col,
        selection_score_margin_rank_limit=selection_score_margin_rank_limit,
        selection_min_score=selection_min_score,
        max_new_names_per_rebalance=max_new_names_per_rebalance,
        max_new_names_shortfall_policy=max_new_names_shortfall_policy,
        max_positive_names=max_positive_names,
    )
    results = _build_position_rows_by_rebalance(setup.context, options)
    return _positions_frame_from_rows(results)
