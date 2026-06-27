from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from market_data_platform.symbols import canonicalize_symbol_columns

DEFAULT_INDUSTRY_COLUMNS = (
    "first_industry_name",
    "industry",
    "sw_l1_name",
    "citic_l1_name",
)
DEFAULT_STYLE_COLUMNS = (
    "size",
    "log_mkt_cap",
    "market_cap",
    "bm",
    "pb",
    "pe_ttm",
    "turnover_rate",
    "volatility_20d",
    "momentum_20d",
    "ret_20d",
)


@dataclass(frozen=True)
class TurnoverAttributionResult:
    summary: dict[str, Any]
    by_window: pd.DataFrame
    by_industry: pd.DataFrame
    by_feature: pd.DataFrame
    by_regime: pd.DataFrame


def compute_turnover_attribution(
    positions_by_rebalance: pd.DataFrame | None,
    scored_data: pd.DataFrame | None = None,
    *,
    feature_importance: pd.DataFrame | None = None,
    feature_columns: Sequence[str] | None = None,
    industry_columns: Sequence[str] = DEFAULT_INDUSTRY_COLUMNS,
    style_columns: Sequence[str] = DEFAULT_STYLE_COLUMNS,
    top_features: int = 10,
) -> TurnoverAttributionResult:
    positions = _normalize_positions(positions_by_rebalance)
    if positions.empty:
        return _empty_result("no_positions")

    scored = _normalize_scored(scored_data)
    industry_col = _first_existing(list(positions.columns), industry_columns)
    if industry_col is None and not scored.empty:
        positions = _merge_scored_columns(positions, scored, list(industry_columns))
        industry_col = _first_existing(list(positions.columns), industry_columns)

    features = _resolve_feature_columns(
        scored,
        feature_importance=feature_importance,
        feature_columns=feature_columns,
        style_columns=style_columns,
        top_features=top_features,
    )
    if features and not scored.empty:
        positions = _merge_scored_columns(positions, scored, features)

    by_window, trade_rows = _window_turnover_rows(positions)
    if trade_rows.empty:
        return TurnoverAttributionResult(
            summary=_summary("insufficient_windows", positions, by_window, trade_rows),
            by_window=by_window,
            by_industry=pd.DataFrame(),
            by_feature=pd.DataFrame(),
            by_regime=pd.DataFrame(),
        )

    by_industry = _industry_attribution(trade_rows, industry_col)
    by_feature = _feature_attribution(trade_rows, features)
    by_regime = _regime_attribution(by_window)
    return TurnoverAttributionResult(
        summary=_summary("ok", positions, by_window, trade_rows),
        by_window=by_window,
        by_industry=by_industry,
        by_feature=by_feature,
        by_regime=by_regime,
    )


def _empty_result(status: str) -> TurnoverAttributionResult:
    return TurnoverAttributionResult(
        summary={"status": status, "windows": 0, "avg_turnover": 0.0},
        by_window=pd.DataFrame(),
        by_industry=pd.DataFrame(),
        by_feature=pd.DataFrame(),
        by_regime=pd.DataFrame(),
    )


def _parse_yyyymmdd(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.strip()
    compact = text.str.replace("-", "", regex=False).str.slice(0, 8)
    parsed = pd.to_datetime(compact, format="%Y%m%d", errors="coerce")
    fallback = pd.to_datetime(values, errors="coerce")
    return parsed.fillna(fallback)


def _normalize_positions(positions: pd.DataFrame | None) -> pd.DataFrame:
    if positions is None or positions.empty:
        return pd.DataFrame()
    required = {"rebalance_date", "symbol", "weight"}
    if not required.issubset(positions.columns):
        return pd.DataFrame()
    out = canonicalize_symbol_columns(positions.copy(), context="turnover attribution positions")
    out["rebalance_date"] = _parse_yyyymmdd(out["rebalance_date"])
    if "entry_date" in out.columns:
        out["entry_date"] = _parse_yyyymmdd(out["entry_date"])
    else:
        out["entry_date"] = out["rebalance_date"]
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce").fillna(0.0)
    rank_values = out["rank"] if "rank" in out.columns else pd.Series(np.nan, index=out.index)
    out["rank"] = pd.to_numeric(rank_values, errors="coerce")
    return out.dropna(subset=["rebalance_date", "symbol"]).copy()


def _normalize_scored(scored: pd.DataFrame | None) -> pd.DataFrame:
    if scored is None or scored.empty or not {"trade_date", "symbol"}.issubset(scored.columns):
        return pd.DataFrame()
    out = canonicalize_symbol_columns(scored.copy(), context="turnover attribution scored data")
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
    return out.dropna(subset=["trade_date", "symbol"]).copy()


def _first_existing(columns: Sequence[str], candidates: Sequence[str]) -> str | None:
    present = set(columns)
    return next((column for column in candidates if column in present), None)


def _merge_scored_columns(
    positions: pd.DataFrame,
    scored: pd.DataFrame,
    columns: Sequence[str],
) -> pd.DataFrame:
    merge_cols = [
        column for column in columns if column in scored.columns and column not in positions.columns
    ]
    if not merge_cols:
        return positions
    supplement = scored[["trade_date", "symbol", *merge_cols]].drop_duplicates(
        subset=["trade_date", "symbol"]
    )
    merged = positions.merge(
        supplement,
        left_on=["rebalance_date", "symbol"],
        right_on=["trade_date", "symbol"],
        how="left",
    ).drop(columns=["trade_date"], errors="ignore")
    missing_cols = [column for column in merge_cols if merged[column].isna().any()]
    if not missing_cols:
        return merged
    return _merge_asof_scored_columns(merged, supplement, missing_cols)


def _merge_asof_scored_columns(
    positions: pd.DataFrame,
    scored: pd.DataFrame,
    columns: Sequence[str],
) -> pd.DataFrame:
    if scored.empty:
        return positions
    out = positions.copy()
    lookup = {
        symbol: group.sort_values("trade_date")
        for symbol, group in scored.groupby("symbol", sort=False)
    }
    for column in columns:
        if column not in scored.columns or column not in out.columns:
            continue
        missing_index = out.index[out[column].isna()]
        if len(missing_index) == 0:
            continue
        for row_index, row in out.loc[
            missing_index,
            ["rebalance_date", "symbol"],
        ].iterrows():
            history = lookup.get(row["symbol"])
            if history is None:
                continue
            prior = history.loc[history["trade_date"] <= row["rebalance_date"], column]
            prior = prior.dropna()
            if not prior.empty:
                out.at[row_index, column] = prior.iloc[-1]
    return out


def _resolve_feature_columns(
    scored: pd.DataFrame,
    *,
    feature_importance: pd.DataFrame | None,
    feature_columns: Sequence[str] | None,
    style_columns: Sequence[str],
    top_features: int,
) -> list[str]:
    candidates: list[str] = []
    if feature_columns:
        candidates.extend(str(column) for column in feature_columns)
    has_feature_source = (
        feature_importance is not None
        and not feature_importance.empty
        and "feature" in feature_importance
    )
    if feature_importance is not None and has_feature_source:
        importance = feature_importance.copy()
        if "importance" in importance:
            importance = importance.sort_values("importance", ascending=False)
        candidates.extend(importance["feature"].astype(str).tolist())
    candidates.extend(style_columns)
    seen: set[str] = set()
    resolved: list[str] = []
    numeric_cols = (
        set(scored.select_dtypes(include=[np.number]).columns) if not scored.empty else set()
    )
    for column in candidates:
        if column in seen or column not in numeric_cols:
            continue
        seen.add(column)
        resolved.append(column)
        if len(resolved) >= top_features:
            break
    return resolved


def _window_turnover_rows(positions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    trades: list[pd.DataFrame] = []
    grouped = list(positions.sort_values("rebalance_date").groupby("rebalance_date", sort=True))
    previous: pd.DataFrame | None = None
    for rebalance_date, current in grouped:
        rebalance_ts = pd.Timestamp(cast(Any, rebalance_date))
        current_weights = current.groupby("symbol", sort=False)["weight"].sum()
        if previous is None:
            previous = current.copy()
            continue
        previous_weights = previous.groupby("symbol", sort=False)["weight"].sum()
        symbols = previous_weights.index.union(current_weights.index)
        prev = previous_weights.reindex(symbols).fillna(0.0)
        curr = current_weights.reindex(symbols).fillna(0.0)
        trade = curr - prev
        abs_trade = trade.abs()
        turnover = float(abs_trade.sum())
        buys = float(trade.clip(lower=0).sum())
        sells = float((-trade.clip(upper=0)).sum())
        entrants = int(((prev == 0) & (curr != 0)).sum())
        exits = int(((prev != 0) & (curr == 0)).sum())
        overlap = int(((prev != 0) & (curr != 0)).sum())
        row: dict[str, Any] = {
            "rebalance_date": rebalance_ts.strftime("%Y%m%d"),
            "entry_date": _format_entry_date(current),
            "turnover": turnover,
            "buy_turnover": buys,
            "sell_turnover": sells,
            "entrant_count": entrants,
            "exit_count": exits,
            "overlap_count": overlap,
            "name_count": int((curr != 0).sum()),
        }
        row["avg_abs_rank_change"] = _avg_abs_rank_change(previous, current)
        rows.append(row)
        trades.append(_trade_frame(current, previous, symbols, prev, curr, trade, rebalance_ts))
        previous = current.copy()
    by_window = pd.DataFrame(rows)
    trade_rows = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame()
    return by_window, trade_rows


def _format_entry_date(current: pd.DataFrame) -> str | None:
    values = pd.to_datetime(current.get("entry_date"), errors="coerce")
    if values.notna().any():
        return pd.Timestamp(values.dropna().iloc[0]).strftime("%Y%m%d")
    return None


def _avg_abs_rank_change(previous: pd.DataFrame, current: pd.DataFrame) -> float:
    if "rank" not in previous.columns or "rank" not in current.columns:
        return float("nan")
    prev_rank = previous.groupby("symbol", sort=False)["rank"].min()
    curr_rank = current.groupby("symbol", sort=False)["rank"].min()
    common = prev_rank.index.intersection(curr_rank.index)
    if len(common) == 0:
        return float("nan")
    delta = (curr_rank.reindex(common) - prev_rank.reindex(common)).abs()
    return float(delta.mean()) if delta.notna().any() else float("nan")


def _trade_frame(
    current: pd.DataFrame,
    previous: pd.DataFrame,
    symbols: pd.Index,
    prev: pd.Series,
    curr: pd.Series,
    trade: pd.Series,
    rebalance_date: pd.Timestamp,
) -> pd.DataFrame:
    current_meta = current.drop_duplicates("symbol").set_index("symbol")
    previous_meta = previous.drop_duplicates("symbol").set_index("symbol")
    meta = current_meta.combine_first(previous_meta).reindex(symbols)
    out = meta.reset_index().rename(columns={"index": "symbol"})
    out["rebalance_date"] = pd.Timestamp(rebalance_date).strftime("%Y%m%d")
    out["previous_weight"] = prev.to_numpy(dtype=float)
    out["current_weight"] = curr.to_numpy(dtype=float)
    out["trade_weight"] = trade.to_numpy(dtype=float)
    out["abs_trade_weight"] = np.abs(out["trade_weight"])
    out["trade_side"] = np.where(out["trade_weight"] >= 0, "buy", "sell")
    return out.loc[out["abs_trade_weight"] > 1e-12].copy()


def _industry_attribution(trades: pd.DataFrame, industry_col: str | None) -> pd.DataFrame:
    if industry_col is None or industry_col not in trades.columns:
        return pd.DataFrame()
    work = trades.copy()
    work["industry"] = work[industry_col].fillna("UNKNOWN").astype(str)
    grouped = work.groupby(["rebalance_date", "industry"], as_index=False)
    return grouped.agg(
        gross_trade=("abs_trade_weight", "sum"),
        net_trade=("trade_weight", "sum"),
        buy_turnover=("trade_weight", lambda x: float(x.clip(lower=0).sum())),
        sell_turnover=("trade_weight", lambda x: float((-x.clip(upper=0)).sum())),
        symbol_count=("symbol", "nunique"),
    ).sort_values(["rebalance_date", "gross_trade"], ascending=[True, False])


def _feature_attribution(trades: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in features:
        if feature not in trades.columns:
            continue
        values = pd.to_numeric(trades[feature], errors="coerce")
        valid = trades.loc[values.notna()].copy()
        if valid.empty:
            continue
        valid["feature_value"] = values.loc[valid.index]
        for rebalance_date, group in valid.groupby("rebalance_date", sort=True):
            weights = group["abs_trade_weight"].astype(float)
            total = float(weights.sum())
            if total <= 0:
                continue
            rows.append(
                {
                    "rebalance_date": rebalance_date,
                    "feature": feature,
                    "trade_weighted_mean": float((group["feature_value"] * weights).sum() / total),
                    "buy_weighted_mean": _side_weighted_mean(group, "buy"),
                    "sell_weighted_mean": _side_weighted_mean(group, "sell"),
                    "gross_trade": total,
                    "symbol_count": int(group["symbol"].nunique()),
                }
            )
    return pd.DataFrame(rows)


def _side_weighted_mean(group: pd.DataFrame, side: str) -> float:
    side_group = group.loc[group["trade_side"] == side]
    if side_group.empty:
        return float("nan")
    weights = side_group["abs_trade_weight"].astype(float)
    total = float(weights.sum())
    if total <= 0:
        return float("nan")
    return float((side_group["feature_value"] * weights).sum() / total)


def _regime_attribution(by_window: pd.DataFrame) -> pd.DataFrame:
    if by_window.empty or "turnover" not in by_window.columns:
        return pd.DataFrame()
    work = by_window.copy()
    median = float(work["turnover"].median())
    work["turnover_regime"] = np.where(work["turnover"] >= median, "high_turnover", "low_turnover")
    work["year"] = pd.to_datetime(work["rebalance_date"], errors="coerce").dt.year
    rows = []
    for keys, group in work.groupby(["year", "turnover_regime"], dropna=False):
        year, regime = keys
        rows.append(
            {
                "year": int(year) if pd.notna(year) else None,
                "turnover_regime": regime,
                "windows": len(group),
                "avg_turnover": float(group["turnover"].mean()),
                "avg_entrant_count": float(group["entrant_count"].mean()),
                "avg_exit_count": float(group["exit_count"].mean()),
                "avg_abs_rank_change": float(group["avg_abs_rank_change"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _mean_or_zero(frame: pd.DataFrame, column: str) -> float:
    if frame.empty:
        return 0.0
    return float(frame[column].mean())


def _summary(
    status: str,
    positions: pd.DataFrame,
    by_window: pd.DataFrame,
    trades: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "status": status,
        "rebalance_count": int(positions["rebalance_date"].nunique()) if not positions.empty else 0,
        "windows": len(by_window),
        "avg_turnover": float(by_window["turnover"].mean()) if not by_window.empty else 0.0,
        "max_turnover": float(by_window["turnover"].max()) if not by_window.empty else 0.0,
        "avg_entrant_count": _mean_or_zero(by_window, "entrant_count"),
        "avg_exit_count": float(by_window["exit_count"].mean()) if not by_window.empty else 0.0,
        "traded_symbols": int(trades["symbol"].nunique()) if not trades.empty else 0,
    }
