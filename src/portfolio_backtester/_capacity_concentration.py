"""Capacity report: concentration of executed notional by stock / liquidity / industry.

Phase 5 groups executed turnover to surface where capacity is consumed. Industry
grouping is opt-in: it is only produced when the positions frame carries an industry
column (passed via ``industry_col``). No external industry source is consulted, keeping
the capacity pipeline free of the heavier exposure-module data dependency.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _liquidity_bucket(liquidity: float, *, quantiles: dict[str, float]) -> str:
    if not np.isfinite(liquidity):
        return "unknown"
    if liquidity <= quantiles.get("p33", float("inf")):
        return "low"
    if liquidity <= quantiles.get("p66", float("inf")):
        return "mid"
    return "high"


def _group_shares(frame: pd.DataFrame, *, group_col: str, value_col: str) -> list[dict[str, Any]]:
    """Aggregate ``value_col`` by ``group_col`` and return shares sorted descending."""
    if frame.empty or group_col not in frame.columns or value_col not in frame.columns:
        return []
    work = frame.copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce").fillna(0.0)
    grouped = work.groupby(group_col, dropna=False)[value_col].sum()
    total = float(grouped.sum())
    if total <= 0.0:
        return []
    rows = [
        {"group": str(key), "executed_notional": float(value), "share": float(value) / total}
        for key, value in grouped.items()
    ]
    rows.sort(key=lambda item: item["executed_notional"], reverse=True)
    return rows


def _hhi(shares: list[dict[str, Any]]) -> float | None:
    if not shares:
        return None
    return float(sum((row["share"] ** 2) for row in shares))


def concentration_by_group(
    *,
    positions: pd.DataFrame,
    pricing: pd.DataFrame,
    executed: Any,
    liquidity_col: str | None = None,
    industry_col: str | None = None,
) -> dict[str, Any]:
    """Phase 5: concentration of executed notional across stock / liquidity / industry.

    Args:
        positions: normalized positions frame (may carry an industry column).
        pricing: normalized pricing frame (may carry a liquidity column).
        executed: adjusted-NAV simulation result exposing ``.fills``.
        liquidity_col: name of the liquidity column in ``pricing``; grouping is produced
            only when present.
        industry_col: name of the industry column in ``positions``; grouping is produced
            only when present.

    Returns a dict with ``by_symbol`` (always), ``by_liquidity`` (when applicable),
    and ``by_industry`` (when applicable), each a list of group shares plus an HHI.
    """
    fills = getattr(executed, "fills", None)
    if fills is None or not isinstance(fills, pd.DataFrame) or fills.empty:
        return {"by_symbol": [], "by_liquidity": None, "by_industry": None}
    fill_frame = fills.copy()
    if "symbol" not in fill_frame.columns or "filled_notional" not in fill_frame.columns:
        return {"by_symbol": [], "by_liquidity": None, "by_industry": None}

    by_symbol = _group_shares(fill_frame, group_col="symbol", value_col="filled_notional")

    by_liquidity: dict[str, Any] | None = None
    if liquidity_col is not None and liquidity_col in pricing.columns:
        liq = pricing[["symbol", liquidity_col]].copy()
        liq[liquidity_col] = pd.to_numeric(liq[liquidity_col], errors="coerce")
        quantiles = {
            "p33": float(liq[liquidity_col].quantile(0.33)),
            "p66": float(liq[liquidity_col].quantile(0.66)),
        }
        liq["liquidity_bucket"] = liq[liquidity_col].apply(
            lambda value: _liquidity_bucket(float(value), quantiles=quantiles)
        )
        merged = fill_frame.merge(liq[["symbol", "liquidity_bucket"]], on="symbol", how="left")
        shares = _group_shares(merged, group_col="liquidity_bucket", value_col="filled_notional")
        by_liquidity = {"groups": shares, "hhi": _hhi(shares)}

    by_industry: dict[str, Any] | None = None
    if industry_col is not None and industry_col in positions.columns:
        ind = positions[["symbol", industry_col]].copy()
        ind = ind[ind[industry_col].notna()]
        ind[industry_col] = ind[industry_col].astype(str)
        merged = fill_frame.merge(ind[["symbol", industry_col]], on="symbol", how="left")
        merged[industry_col] = merged[industry_col].fillna("unknown")
        shares = _group_shares(merged, group_col=industry_col, value_col="filled_notional")
        by_industry = {"groups": shares, "hhi": _hhi(shares)}

    return {
        "by_symbol": by_symbol,
        "by_symbol_hhi": _hhi(by_symbol),
        "by_liquidity": by_liquidity,
        "by_industry": by_industry,
    }
