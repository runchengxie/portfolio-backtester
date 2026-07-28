"""Portfolio position construction: context, dataclasses, and rebalance setup."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .execution import ExecutionModel, SelectionConstraints
from .execution_calendar import build_execution_date_map
from .portfolio_position_frames import resolve_pricing_source
from .portfolio_weights import normalize_weighting_mode
from .selection_controls import (
    apply_liquidity_floor_to_day as _apply_liquidity_floor_to_day,
    merge_pricing_supplemental_columns as _merge_pricing_supplemental_columns,
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
    pricing_source = resolve_pricing_source(data, pricing_data)
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
