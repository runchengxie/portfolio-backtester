from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from ._exposure_columns import (
    _STYLE_FACTOR_ORDER,
    _build_benchmark_daily_returns,
    _build_beta_table,
    _empty_exposure_result,
    _exposure_period_key,
    _price_history_tables,
)
from ._exposure_industry import (
    _apply_industry_labels_asof,
    _finalize_industry_exposure,
    _industry_exposure_rows,
    _normalize_exposure_industry_source,
    _normalize_exposure_positions,
    _normalize_exposure_scored_data,
    _resolve_exposure_industry_context,
)
from ._exposure_style import (
    _finalize_style_exposure,
    _style_rows_for_rebalance,
)


def _build_active_exposure_summary_table(
    style_df: pd.DataFrame,
    industry_df: pd.DataFrame,
    *,
    top_n_industries: int = 3,
) -> pd.DataFrame:
    if style_df.empty and industry_df.empty:
        return pd.DataFrame()

    style_work, industry_work = style_df.copy(), industry_df.copy()
    for frame in (style_work, industry_work):
        if not frame.empty:
            frame["rebalance_date"] = frame["rebalance_date"].map(_exposure_period_key)
            frame["entry_date"] = frame["entry_date"].map(_exposure_period_key)
    periods: set[tuple[str, str | None]] = set()
    for frame in (style_work, industry_work):
        if frame.empty:
            continue
        for _, row in frame[["rebalance_date", "entry_date"]].drop_duplicates().iterrows():
            periods.add((str(row["rebalance_date"]), row["entry_date"]))

    rows: list[dict[str, Any]] = []
    for rebalance_date, entry_date in sorted(periods):
        row: dict[str, Any] = {
            "rebalance_date": rebalance_date,
            "entry_date": entry_date,
        }
        if not style_work.empty:
            style_day = style_work[style_work["rebalance_date"] == rebalance_date]
            for factor in _STYLE_FACTOR_ORDER:
                factor_day = style_day[style_day["factor"] == factor]
                if factor_day.empty:
                    continue
                factor_row = factor_day.iloc[0]
                row[f"{factor}_active_net_vs_equal"] = factor_row["active_net_vs_equal"]
                row[f"{factor}_active_net_vs_cap"] = factor_row["active_net_vs_cap"]
                row[f"{factor}_weight_coverage"] = factor_row["weight_coverage"]
                row[f"{factor}_source"] = factor_row["source"]

        if not industry_work.empty:
            industry_day = industry_work[industry_work["rebalance_date"] == rebalance_date].copy()
            if not industry_day.empty:
                row["industry_column"] = industry_day["industry_col"].dropna().iloc[0]
                reference_col = (
                    "active_net_vs_cap_weight"
                    if industry_day["active_net_vs_cap_weight"].notna().any()
                    else "active_net_vs_equal_weight"
                )
                row["industry_reference"] = reference_col
                ranked = industry_day.assign(
                    abs_active=industry_day[reference_col].abs()
                ).sort_values(["abs_active", "industry"], ascending=[False, True])
                top_industries = ranked.head(top_n_industries).iterrows()
                for idx, (_, ranked_row) in enumerate(top_industries, start=1):
                    row[f"industry_top_{idx}_name"] = ranked_row["industry"]
                    row[f"industry_top_{idx}_active"] = ranked_row[reference_col]
                    row[f"industry_top_{idx}_portfolio_net_weight"] = ranked_row[
                        "portfolio_net_weight"
                    ]

        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_df.sort_values("rebalance_date", inplace=True)
    summary_df.reset_index(drop=True, inplace=True)
    return summary_df


def _build_exposure_rows(
    *,
    scored: pd.DataFrame,
    positions: pd.DataFrame,
    history: Mapping[str, Any],
    beta_table: pd.DataFrame,
    industry_col: str | None,
    industry_history: Mapping[str, tuple[np.ndarray, np.ndarray]],
    market_cap_col: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    style_rows: list[dict[str, Any]] = []
    industry_rows: list[dict[str, Any]] = []
    factor_meta: dict[str, dict[str, Any]] = {}
    by_date = scored.groupby("trade_date", sort=True)
    for rebalance_date, pos_day in positions.groupby("rebalance_date_ts", sort=True):
        if rebalance_date not in by_date.groups:
            continue
        day = by_date.get_group(rebalance_date).copy()
        entry_date = pos_day["entry_date_ts"].iloc[0] if not pos_day.empty else None
        style_rows.extend(
            _style_rows_for_rebalance(
                day=day,
                positions=pos_day,
                rebalance_date=rebalance_date,
                entry_date=entry_date,
                market_cap_col=market_cap_col,
                history=history,
                beta_table=beta_table,
                factor_meta=factor_meta,
            )
        )
        if industry_col is not None:
            industry_day = _apply_industry_labels_asof(
                day,
                industry_col=industry_col,
                rebalance_date=rebalance_date,
                industry_history=industry_history,
            )
            industry_rows.extend(
                _industry_exposure_rows(
                    positions=pos_day,
                    day=industry_day,
                    industry_col=industry_col,
                    market_cap_col=market_cap_col,
                    rebalance_date=rebalance_date,
                    entry_date=entry_date,
                )
            )
    return style_rows, industry_rows, factor_meta


def compute_backtest_exposure_analysis(
    scored_data: pd.DataFrame | None,
    positions_by_rebalance: pd.DataFrame | None,
    *,
    pricing_data: pd.DataFrame | None = None,
    price_col: str = "close",
    benchmark_df: pd.DataFrame | None = None,
    benchmark_return_series: pd.Series | None = None,
    market_cap_col: str | None = None,
    industry_columns: Sequence[str] | None = None,
    industry_source_data: pd.DataFrame | None = None,
) -> dict[str, Any]:
    if (
        scored_data is None
        or scored_data.empty
        or positions_by_rebalance is None
        or positions_by_rebalance.empty
    ):
        return _empty_exposure_result()

    scored = _normalize_exposure_scored_data(scored_data)
    industry_source = _normalize_exposure_industry_source(industry_source_data)
    positions = _normalize_exposure_positions(positions_by_rebalance)
    if positions.empty:
        return _empty_exposure_result()

    history = _price_history_tables(pricing_data, price_col=price_col)
    benchmark_returns = _build_benchmark_daily_returns(
        benchmark_df,
        benchmark_return_series,
        price_col=price_col,
    )
    beta_table = _build_beta_table(history["returns"], benchmark_returns)
    industry_col, industry_history = _resolve_exposure_industry_context(
        scored,
        industry_source,
        industry_columns=industry_columns,
    )
    style_rows, industry_rows, factor_meta = _build_exposure_rows(
        scored=scored,
        positions=positions,
        history=history,
        beta_table=beta_table,
        industry_col=industry_col,
        industry_history=industry_history,
        market_cap_col=market_cap_col,
    )
    style_df, style_summary = _finalize_style_exposure(style_rows, factor_meta)
    industry_df, industry_summary = _finalize_industry_exposure(industry_rows, industry_col)
    return {
        "style": style_df,
        "style_summary": style_summary,
        "industry": industry_df,
        "industry_summary": industry_summary,
        "active_summary": _build_active_exposure_summary_table(style_df, industry_df),
    }
