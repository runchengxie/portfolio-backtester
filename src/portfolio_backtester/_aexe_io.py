"""A-share executable OOS Top-K: global-free I/O and statistic helpers.

These helpers do not read the module-level configuration globals, so they are
safe to live in a private submodule. Re-exported by
:mod:`portfolio_backtester.a_share_executable_oos_topk`.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CACHE_FILE_TEMPLATE = (
    "a_share_tushare_a_share_pit_top800_2015_weekly_"
    "three_statement_core_probe_daily_{symbol}.parquet"
)


def _date8(value: Any) -> str:
    text = str(value)
    if "-" in text:
        return pd.to_datetime(text).strftime("%Y%m%d")
    return text[:8]


def load_positions(run_dir: Path) -> pd.DataFrame:
    pos = pd.read_csv(run_dir / "positions_by_rebalance_oos.csv")
    pos["rebalance_date"] = pos["rebalance_date"].map(_date8)
    pos["entry_date"] = pos["entry_date"].map(_date8)
    pos["symbol"] = pos["symbol"].astype(str)
    pos["rank"] = pd.to_numeric(pos["rank"], errors="coerce")
    pos = pos.dropna(subset=["rank"])
    return pos.sort_values(["rebalance_date", "rank", "symbol"])


def load_prices(symbols: list[str], run_dir: Path) -> pd.DataFrame:
    frames = []
    cache = run_dir.parents[1] / "cache"
    for sym in symbols:
        path = cache / CACHE_FILE_TEMPLATE.format(symbol=sym)
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        keep = [
            c
            for c in [
                "trade_date",
                "symbol",
                "tr_close",
                "amount",
                "medadv20_amount",
                "is_tradable",
                "is_suspended",
                "is_limit_up",
                "is_limit_down",
                "up_limit",
                "down_limit",
            ]
            if c in df.columns
        ]
        df = df[keep].copy()
        df["trade_date"] = df["trade_date"].map(_date8)
        df["symbol"] = df["symbol"].astype(str)
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No daily price cache files found for OOS symbols")
    return (
        pd.concat(frames, ignore_index=True)
        .dropna(subset=["trade_date", "symbol", "tr_close"])
        .drop_duplicates(["trade_date", "symbol"], keep="last")
    )


def portfolio_value(holdings: dict[str, int], prices: pd.Series, cash: float) -> float:
    value = cash
    for sym, qty in holdings.items():
        px = prices.get(sym, np.nan)
        if pd.notna(px) and px > 0:
            value += float(px) * qty
    return float(value)


def _rank_map(candidates: pd.DataFrame) -> dict[str, int]:
    return {str(row["symbol"]): int(row["rank"]) for _, row in candidates.iterrows()}


def _trade_notional(
    holdings: dict[str, int], target_alloc: dict[str, int], prices: pd.Series
) -> float:
    traded = 0.0
    for sym in sorted(set(holdings) | set(target_alloc)):
        pxv = prices.get(sym, np.nan)
        if pd.isna(pxv) or pxv <= 0:
            continue
        delta = target_alloc.get(sym, 0) - holdings.get(sym, 0)
        traded += abs(delta) * float(pxv)
    return traded


def _row_value(row: pd.Series, name: str, default: Any = np.nan) -> Any:
    return row.get(name, default) if name in row.index else default


def _adv_notional(row: pd.Series) -> float:
    value = _row_value(row, "medadv20_amount", np.nan)
    if pd.isna(value) or float(value) <= 0:
        value = _row_value(row, "amount", np.nan)
    if pd.isna(value) or float(value) <= 0:
        return np.nan
    # Tushare daily amount is conventionally reported in thousand CNY.
    return float(value) * 1000.0


def _adv_bucket(adv_notional: float) -> str:
    if not math.isfinite(adv_notional) or adv_notional <= 0:
        return "missing"
    if adv_notional < 10_000_000:
        return "lt_10m"
    if adv_notional < 50_000_000:
        return "10m_50m"
    if adv_notional < 200_000_000:
        return "50m_200m"
    return "gte_200m"


def _turnover_action_order(
    holdings: dict[str, int], target_alloc: dict[str, int], prices: pd.Series
) -> list[tuple[int, str, int, float]]:
    actions = []
    for sym in sorted(set(holdings) | set(target_alloc)):
        pxv = prices.get(sym, np.nan)
        if pd.isna(pxv) or pxv <= 0:
            continue
        delta = target_alloc.get(sym, 0) - holdings.get(sym, 0)
        if delta == 0:
            continue
        priority = 0 if delta < 0 and target_alloc.get(sym, 0) == 0 else 1
        priority = 2 if delta > 0 else priority
        actions.append((priority, sym, delta, float(pxv)))
    return sorted(actions)


def _holding_values(holdings: dict[str, int], prices: pd.Series) -> list[float]:
    return [
        float(prices.get(s, np.nan)) * q
        for s, q in holdings.items()
        if pd.notna(prices.get(s, np.nan))
    ]


def _market_rows_by_symbol(px: pd.DataFrame, date: str) -> dict[str, pd.Series]:
    rows = px[px["trade_date"] == date]
    return {str(row["symbol"]): row for _, row in rows.iterrows()}


def _blocked_trade_count(trades: pd.DataFrame) -> int:
    if trades.empty or "blocked_or_capped" not in trades.columns:
        return 0
    return int(trades["blocked_or_capped"].fillna(False).sum())


def _avg_impact_bps(trades: pd.DataFrame) -> float:
    if trades.empty or "impact_bps" not in trades.columns:
        return 0.0
    active = trades[pd.to_numeric(trades["notional"], errors="coerce").fillna(0.0) > 0]
    if active.empty:
        return 0.0
    return float(pd.to_numeric(active["impact_bps"], errors="coerce").fillna(0.0).mean())


def compute_stats(daily: pd.DataFrame) -> dict[str, Any]:
    r = pd.to_numeric(daily["daily_return"], errors="coerce").fillna(0.0)
    nav = pd.to_numeric(daily["nav"], errors="coerce").ffill()
    years = len(daily) / 252.0
    total = float(nav.iloc[-1] - 1.0)
    ann = float(nav.iloc[-1] ** (1 / years) - 1) if years > 0 and nav.iloc[-1] > 0 else np.nan
    vol = float(r.std(ddof=1) * np.sqrt(252)) if len(r) > 1 else np.nan
    sharpe = (
        float(r.mean() / r.std(ddof=1) * np.sqrt(252))
        if len(r) > 1 and r.std(ddof=1) > 0
        else np.nan
    )
    dd = nav / nav.cummax() - 1.0
    roll63 = r.rolling(63).mean() / r.rolling(63).std(ddof=1) * np.sqrt(252)
    roll126 = r.rolling(126).mean() / r.rolling(126).std(ddof=1) * np.sqrt(252)
    return {
        "daily_rows": len(daily),
        "start": str(daily["trade_date"].iloc[0]),
        "end": str(daily["trade_date"].iloc[-1]),
        "total_return": total,
        "ann_return": ann,
        "ann_vol": vol,
        "sharpe": sharpe,
        "max_drawdown": float(dd.min()),
        "rolling_sharpe_3m_last": float(roll63.dropna().iloc[-1])
        if roll63.notna().any()
        else np.nan,
        "rolling_sharpe_6m_last": float(roll126.dropna().iloc[-1])
        if roll126.notna().any()
        else np.nan,
    }
