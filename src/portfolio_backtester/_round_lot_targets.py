"""Round-lot target selection with liquidity, prior-holding, and industry caps."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd

from ._round_lot_weights import RoundLotVariant, _numeric_series, cap_and_redistribute


def select_round_lot_targets(
    day: pd.DataFrame,
    variant: RoundLotVariant,
    previous_symbols: set[str] | None = None,
) -> pd.DataFrame:
    """Select target names with liquidity, prior-holding, and industry-cap constraints."""
    if day.empty:
        return pd.DataFrame()
    required = {"symbol", "signal_backtest", "medadv20_amount"}
    missing = sorted(required - set(day.columns))
    if missing:
        raise ValueError(f"round-lot target selection missing required columns: {missing}")

    previous_symbols = previous_symbols or set()
    frame = day.copy()
    liquidity = _numeric_series(frame["medadv20_amount"], index=frame.index)
    if variant.liquidity_floor_q > 0 and int(liquidity.notna().sum()) > 1:
        frame = frame[
            (liquidity.isna()) | (liquidity >= liquidity.quantile(variant.liquidity_floor_q))
        ]
        frame = frame.copy()
    frame = cast("pd.DataFrame", cast(Any, frame).sort_values("signal_backtest", ascending=False))
    if "first_industry_name" not in frame.columns:
        frame["first_industry_name"] = "UNKNOWN"

    keep_pool = frame.head(max(variant.target_holdings * 2, variant.target_holdings))
    keep_symbols = set(
        keep_pool.loc[keep_pool["symbol"].isin(list(previous_symbols)), "symbol"].astype(str)
    )
    ordered = pd.concat(
        [
            frame[frame["symbol"].isin(list(keep_symbols))],
            frame[~frame["symbol"].isin(list(keep_symbols))],
        ],
        ignore_index=True,
    )
    ordered = cast("pd.DataFrame", cast(Any, ordered).drop_duplicates("symbol", keep="first"))

    selected: list[pd.Series] = []
    industry_counts: dict[str, int] = {}
    for _, row in ordered.iterrows():
        industry = str(row.get("first_industry_name") or "UNKNOWN")
        if industry_counts.get(industry, 0) >= variant.industry_cap:
            continue
        selected.append(cast("pd.Series", row))
        industry_counts[industry] = industry_counts.get(industry, 0) + 1
        if len(selected) >= variant.target_holdings:
            break
    if not selected:
        return pd.DataFrame()

    out = pd.DataFrame(selected).copy()
    if variant.weighting == "equal":
        raw = pd.Series(1.0, index=out.index)
    else:
        liquidity_raw = _numeric_series(out["medadv20_amount"], index=out.index).clip(lower=1.0)
        raw = pd.Series(np.sqrt(liquidity_raw.to_numpy()), index=out.index)
    out["target_weight"] = cap_and_redistribute(raw, variant.max_weight).to_numpy()
    return out
