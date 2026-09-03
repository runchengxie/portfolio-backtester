"""Pure selection helpers for converting saved positions into allocations."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd


def _ensure_symbol_column(df: pd.DataFrame, *, context: str) -> pd.DataFrame:
    out = df.copy()
    candidates = [
        column for column in ("symbol", "ts_code", "stock_ticker", "order_book_id") if column in out
    ]
    if not candidates:
        raise SystemExit(f"{context} is missing symbol/stock_ticker/ts_code/order_book_id.")
    merged = out[candidates[0]].where(out[candidates[0]].notna(), "").astype(str).str.strip()
    for column in candidates[1:]:
        values = out[column].where(out[column].notna(), "").astype(str).str.strip()
        merged = merged.where(merged != "", values)
    out["symbol"] = merged
    return out


def select_from_positions_file(
    positions_path: Path,
    as_of: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    if not positions_path.exists():
        raise SystemExit(f"Positions file not found: {positions_path}")
    df = pd.read_csv(positions_path)
    if df.empty:
        raise SystemExit(f"{positions_path.name} is empty.")
    if "entry_date" not in df.columns:
        raise SystemExit(f"{positions_path.name} is missing entry_date.")
    entry_dates = pd.to_datetime(df["entry_date"], errors="coerce").dt.normalize()
    if entry_dates.isna().all():
        raise SystemExit("Failed to parse entry_date column.")
    eligible = entry_dates <= as_of
    if not eligible.any():
        raise SystemExit("No holdings available before the requested --as-of date.")
    latest_entry = cast(pd.Timestamp, entry_dates[eligible].max())
    selection = cast(pd.DataFrame, df[entry_dates == latest_entry].copy())
    selection = _ensure_symbol_column(selection, context=positions_path.name)
    selection["symbol"] = selection["symbol"].astype(str).str.strip()
    if selection.empty:
        raise SystemExit("No holdings found for the latest entry date.")
    return selection, latest_entry


def prepare_selection(
    selection: pd.DataFrame,
    *,
    side: str,
    top_n: int,
) -> pd.DataFrame:
    prepared = _ensure_symbol_column(selection, context="Holdings payload")
    prepared = prepared.drop(columns=["order_book_id"], errors="ignore")
    prepared["symbol"] = prepared["symbol"].astype(str).str.strip()
    if "side" not in prepared.columns:
        prepared["side"] = "long"
    if "rank" not in prepared.columns:
        prepared["rank"] = np.nan
    prepared["side"] = prepared["side"].astype(str).str.lower()
    if side != "all":
        prepared = prepared[prepared["side"] == side].copy()
    if prepared.empty:
        raise SystemExit(f"No holdings available for --side={side}.")
    prepared = (
        prepared.sort_values(
            by=["side", "rank", "symbol"],
            na_position="last",
        )
        .head(top_n)
        .copy()
    )
    if prepared.empty:
        raise SystemExit("No holdings available after --top-n filtering.")
    prepared.reset_index(drop=True, inplace=True)
    return cast(pd.DataFrame, prepared)
