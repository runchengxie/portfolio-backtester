"""Write derived position views for a backtest run directory."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ._symbol_utils import canonicalize_symbol_columns


def _coerce_yyyymmdd(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.strip()
    compact = text.str.replace("-", "", regex=False)
    parsed = pd.to_datetime(compact, format="%Y%m%d", errors="coerce")
    formatted = parsed.dt.strftime("%Y%m%d")
    return formatted.where(parsed.notna(), text)


def _save_frame(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _build_rebalance_diff(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "entry_date" not in frame.columns:
        return pd.DataFrame()
    normalized = canonicalize_symbol_columns(frame, context="Rebalance diff")
    entry_compact = _coerce_yyyymmdd(normalized["entry_date"])
    entry_dt = pd.to_datetime(entry_compact, format="%Y%m%d", errors="coerce")
    unique_entries = sorted(entry_dt.dropna().unique())
    if len(unique_entries) < 2:
        return pd.DataFrame()

    latest_entry, previous_entry = unique_entries[-1], unique_entries[-2]
    current = normalized[entry_dt == latest_entry].copy()
    previous = normalized[entry_dt == previous_entry].copy()
    for positions in (current, previous):
        if "side" not in positions.columns:
            positions["side"] = "long"
        if "weight" not in positions.columns:
            positions["weight"] = np.nan
        if "signal" not in positions.columns:
            positions["signal"] = np.nan
        if "rank" not in positions.columns:
            positions["rank"] = np.nan

    current = current[["symbol", "side", "weight", "signal", "rank"]]
    previous = previous[["symbol", "side", "weight", "signal", "rank"]].rename(
        columns={"weight": "weight_prev", "signal": "signal_prev", "rank": "rank_prev"}
    )
    merged = current.merge(previous, on=["symbol", "side"], how="outer", indicator=True)
    merged["weight"] = merged["weight"].fillna(0.0)
    merged["weight_prev"] = merged["weight_prev"].fillna(0.0)
    merged["weight_delta"] = merged["weight"] - merged["weight_prev"]
    merged["change"] = (
        merged["_merge"]
        .astype(str)
        .map({"left_only": "added", "right_only": "removed", "both": "changed"})
    )
    merged.loc[(merged["_merge"] == "both") & (merged["weight_delta"].abs() < 1e-12), "change"] = (
        "unchanged"
    )
    merged = merged[merged["change"] != "unchanged"].copy()
    merged["entry_date"] = latest_entry.strftime("%Y%m%d")
    merged["entry_date_prev"] = previous_entry.strftime("%Y%m%d")
    merged.drop(columns=["_merge"], inplace=True)
    merged.sort_values(["change", "side", "symbol"], inplace=True)
    return canonicalize_symbol_columns(merged, context="Rebalance diff")


def write_position_outputs(
    *,
    positions: pd.DataFrame | None,
    run_dir: Path,
    by_rebalance_name: str,
    current_name: str,
    diff_name: str,
    artifacts: dict[str, Any],
    by_rebalance_key: str,
    current_key: str,
    diff_key: str,
    enabled: bool,
) -> None:
    """Write positions, latest positions, and the latest rebalance diff."""
    if positions is None or positions.empty or not enabled:
        return

    by_rebalance_path = run_dir / by_rebalance_name
    _save_frame(positions, by_rebalance_path)
    artifacts[by_rebalance_key] = by_rebalance_path

    entry_dates = pd.to_datetime(
        _coerce_yyyymmdd(positions["entry_date"]), format="%Y%m%d", errors="coerce"
    )
    if entry_dates.notna().any():
        current = positions.loc[entry_dates == entry_dates.max()].copy()
        if not current.empty:
            current_path = run_dir / current_name
            _save_frame(current, current_path)
            artifacts[current_key] = current_path

    diff = _build_rebalance_diff(positions)
    if not diff.empty:
        diff_path = run_dir / diff_name
        _save_frame(diff, diff_path)
        artifacts[diff_key] = diff_path
