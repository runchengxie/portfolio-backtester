from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from portfolio_backtester._symbol_utils import canonicalize_symbol_columns

from ._exposure_columns import (
    _as_numeric,
    _clean_categorical_labels,
    _empty_industry_summary,
    _resolve_industry_column,
    _to_datetime_series,
)


def _normalize_exposure_scored_data(scored_data: pd.DataFrame) -> pd.DataFrame:
    scored = scored_data.copy()
    scored = canonicalize_symbol_columns(scored, context="Exposure scored data")
    scored["trade_date"] = pd.to_datetime(scored["trade_date"], errors="coerce").dt.normalize()
    scored = scored.dropna(subset=["trade_date", "symbol"])
    return scored.drop_duplicates(subset=["trade_date", "symbol"], keep="last")


def _normalize_exposure_industry_source(
    industry_source_data: pd.DataFrame | None,
) -> pd.DataFrame:
    if industry_source_data is None or industry_source_data.empty:
        return pd.DataFrame()
    industry_source = industry_source_data.copy()
    industry_source = canonicalize_symbol_columns(
        industry_source,
        context="Exposure industry source",
    )
    industry_source["trade_date"] = pd.to_datetime(
        industry_source["trade_date"], errors="coerce"
    ).dt.normalize()
    industry_source = industry_source.dropna(subset=["trade_date", "symbol"])
    return industry_source.drop_duplicates(subset=["trade_date", "symbol"], keep="last")


def _normalize_exposure_positions(positions_by_rebalance: pd.DataFrame) -> pd.DataFrame:
    positions = positions_by_rebalance.copy()
    positions = canonicalize_symbol_columns(positions, context="Exposure positions")
    positions["rebalance_date_ts"] = _to_datetime_series(positions["rebalance_date"])
    positions["entry_date_ts"] = _to_datetime_series(positions["entry_date"])
    return positions.dropna(subset=["rebalance_date_ts", "entry_date_ts", "symbol"])


def _build_industry_history(
    frame: pd.DataFrame,
    *,
    industry_col: str,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    history = frame[["symbol", "trade_date", industry_col]].copy()
    history[industry_col] = _clean_categorical_labels(history[industry_col])
    history = history.dropna(subset=["symbol", "trade_date", industry_col])
    if history.empty:
        return {}
    history = history.sort_values(["symbol", "trade_date"]).drop_duplicates(
        subset=["symbol", "trade_date"],
        keep="last",
    )

    by_symbol: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for symbol, group in history.groupby("symbol", sort=False):
        dates = group["trade_date"].to_numpy(dtype="datetime64[ns]")
        labels = group[industry_col].to_numpy(dtype=object)
        if len(dates) == 0:
            continue
        by_symbol[str(symbol)] = (dates, labels)
    return by_symbol


def _apply_industry_labels_asof(
    day: pd.DataFrame,
    *,
    industry_col: str,
    rebalance_date: pd.Timestamp,
    industry_history: Mapping[str, tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    work = day.copy()
    if industry_col in work.columns:
        industry_values = _clean_categorical_labels(work[industry_col])
    else:
        industry_values = pd.Series(pd.NA, index=work.index, dtype="object")

    missing = industry_values.isna()
    if missing.any():
        rebalance_dt64 = rebalance_date.to_datetime64()
        for idx, symbol in work.loc[missing, "symbol"].items():
            history = industry_history.get(str(symbol))
            if history is None:
                continue
            dates, labels = history
            pos = int(np.searchsorted(dates, rebalance_dt64, side="right") - 1)
            if pos >= 0:
                industry_values.at[idx] = labels[pos]

    work[industry_col] = industry_values
    return work


def _resolve_exposure_industry_context(
    scored: pd.DataFrame,
    industry_source: pd.DataFrame,
    *,
    industry_columns: Sequence[str] | None,
) -> tuple[str | None, Mapping[str, tuple[np.ndarray, np.ndarray]]]:
    industry_col = _resolve_industry_column(scored, industry_columns=industry_columns)
    if industry_col is None and not industry_source.empty:
        industry_col = _resolve_industry_column(
            industry_source,
            industry_columns=industry_columns,
        )
    industry_history_source = (
        industry_source
        if industry_col is not None and industry_col in industry_source.columns
        else scored
    )
    industry_history = (
        _build_industry_history(industry_history_source, industry_col=industry_col)
        if industry_col is not None
        else {}
    )
    return industry_col, industry_history


def _industry_exposure_rows(
    *,
    positions: pd.DataFrame,
    day: pd.DataFrame,
    industry_col: str,
    market_cap_col: str | None,
    rebalance_date: pd.Timestamp,
    entry_date: pd.Timestamp | None,
) -> list[dict[str, Any]]:
    universe = day[["symbol", industry_col]].drop_duplicates(subset=["symbol"], keep="last").copy()
    universe[industry_col] = _clean_categorical_labels(universe[industry_col])
    universe = universe.dropna(subset=[industry_col]).copy()
    if universe.empty:
        return []

    pos = positions.groupby("symbol", as_index=False)["weight"].sum()
    pos = pos.merge(universe, on="symbol", how="inner")

    long = pos.loc[pos["weight"] > 0].copy()
    short = pos.loc[pos["weight"] < 0].copy()
    short["weight"] = short["weight"].abs()
    gross = pos.copy()
    gross["weight"] = gross["weight"].abs()

    long_total = float(long["weight"].sum()) if not long.empty else 0.0
    short_total = float(short["weight"].sum()) if not short.empty else 0.0
    gross_total = float(gross["weight"].sum()) if not gross.empty else 0.0

    long_share = (
        long.groupby(industry_col)["weight"].sum() / long_total
        if long_total > 0
        else pd.Series(dtype=float)
    )
    short_share = (
        short.groupby(industry_col)["weight"].sum() / short_total
        if short_total > 0
        else pd.Series(dtype=float)
    )
    gross_share = (
        gross.groupby(industry_col)["weight"].sum() / gross_total
        if gross_total > 0
        else pd.Series(dtype=float)
    )
    universe_equal = universe.groupby(industry_col)["symbol"].nunique()
    universe_equal = (
        universe_equal / float(universe_equal.sum()) if not universe_equal.empty else universe_equal
    )

    universe_cap = pd.Series(dtype=float)
    if market_cap_col and market_cap_col in day.columns:
        cap = (
            day[["symbol", industry_col, market_cap_col]]
            .drop_duplicates(subset=["symbol"], keep="last")
            .copy()
        )
        cap[market_cap_col] = _as_numeric(cap[market_cap_col])
        cap = cap.loc[cap[market_cap_col] > 0]
        cap[industry_col] = _clean_categorical_labels(cap[industry_col])
        cap = cap.dropna(subset=[industry_col])
        if not cap.empty:
            universe_cap = cap.groupby(industry_col)[market_cap_col].sum()
            universe_cap = universe_cap / float(universe_cap.sum())

    industries = sorted(
        set(universe_equal.index)
        | set(long_share.index)
        | set(short_share.index)
        | set(gross_share.index)
        | set(universe_cap.index)
    )
    rows: list[dict[str, Any]] = []
    for industry in industries:
        long_weight = float(long_share.get(industry, 0.0))
        short_weight = float(short_share.get(industry, 0.0))
        gross_weight = float(gross_share.get(industry, 0.0))
        net_weight = float(long_weight - short_weight)
        equal_weight = float(universe_equal.get(industry, 0.0))
        cap_weight = float(universe_cap.get(industry, np.nan))
        rows.append(
            {
                "rebalance_date": rebalance_date.strftime("%Y%m%d"),
                "entry_date": entry_date.strftime("%Y%m%d") if entry_date is not None else None,
                "industry": str(industry),
                "industry_col": industry_col,
                "portfolio_long_weight": long_weight,
                "portfolio_short_weight": short_weight,
                "portfolio_net_weight": net_weight,
                "portfolio_gross_weight": gross_weight,
                "universe_equal_weight": equal_weight,
                "universe_cap_weight": cap_weight,
                "active_net_vs_equal_weight": float(net_weight - equal_weight),
                "active_net_vs_cap_weight": (
                    float(net_weight - cap_weight) if np.isfinite(cap_weight) else np.nan
                ),
            }
        )
    return rows


def _finalize_industry_exposure(
    industry_rows: list[dict[str, Any]],
    industry_col: str | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    industry_df = pd.DataFrame(industry_rows)
    if not industry_df.empty:
        industry_df.sort_values(["rebalance_date", "industry"], inplace=True)
        industry_df.reset_index(drop=True, inplace=True)
    industry_summary = _empty_industry_summary()
    industry_summary["industry_column"] = industry_col
    if industry_df.empty:
        return industry_df, industry_summary
    latest_rebalance = str(industry_df["rebalance_date"].max())
    latest_industry = industry_df[industry_df["rebalance_date"] == latest_rebalance].copy()
    industry_summary["latest_rebalance_date"] = latest_rebalance
    latest_entry = latest_industry["entry_date"].dropna()
    industry_summary["latest_entry_date"] = (
        str(latest_entry.iloc[0]) if not latest_entry.empty else None
    )
    reference_col = (
        "active_net_vs_cap_weight"
        if latest_industry["active_net_vs_cap_weight"].notna().any()
        else "active_net_vs_equal_weight"
    )
    latest_industry["abs_active"] = latest_industry[reference_col].abs()
    latest_industry = latest_industry.sort_values("abs_active", ascending=False)
    industry_summary["latest"] = {
        "reference": reference_col,
        "top_absolute_active": [
            {
                "industry": str(row["industry"]),
                "portfolio_net_weight": float(row["portfolio_net_weight"]),
                "universe_equal_weight": float(row["universe_equal_weight"]),
                "universe_cap_weight": (
                    float(row["universe_cap_weight"])
                    if pd.notna(row["universe_cap_weight"])
                    else np.nan
                ),
                "active_net_vs_equal_weight": float(row["active_net_vs_equal_weight"]),
                "active_net_vs_cap_weight": (
                    float(row["active_net_vs_cap_weight"])
                    if pd.notna(row["active_net_vs_cap_weight"])
                    else np.nan
                ),
            }
            for _, row in latest_industry.head(10).iterrows()
        ],
    }
    return industry_df, industry_summary
