"""Position backtest configuration and input normalization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from .contracts import assert_positions_by_rebalance_frame

PositionExitPolicy = Literal["period", "strict", "ffill", "delay"]


@dataclass(frozen=True)
class PositionBacktestConfig:
    price_col: str = "close"
    entry_price_col: str | None = None
    exit_price_col: str | None = None
    transaction_cost_bps: float = 0.0
    trading_days_per_year: int = 252
    long_only: bool = True
    preserve_gross_exposure: bool = False
    exit_price_policy: PositionExitPolicy = "period"
    exit_fallback_policy: Literal["ffill", "none"] = "ffill"
    tradable_col: str | None = None

    @property
    def effective_entry_price_col(self) -> str:
        return self.entry_price_col or self.price_col

    @property
    def effective_exit_price_col(self) -> str:
        return self.exit_price_col or self.price_col


@dataclass(frozen=True)
class PositionBacktestResult:
    net_returns: pd.DataFrame
    gross_returns: pd.DataFrame
    periods: pd.DataFrame
    summary: dict[str, Any]


def _date_series(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    parsed = pd.to_datetime(text, errors="coerce")
    compact = text.str.fullmatch(r"\d{8}")
    if compact.any():
        parsed.loc[compact] = pd.to_datetime(text.loc[compact], format="%Y%m%d", errors="coerce")
    return parsed.dt.normalize()


def _date_value(value: Any) -> pd.Timestamp:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if len(text) == 8 and text.isdigit():
        parsed = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    else:
        parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return pd.NaT
    return pd.Timestamp(parsed).normalize()


def _date_key(value: Any) -> str:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return str(value).strip().replace("-", "").replace(".0", "")
    return pd.Timestamp(timestamp).strftime("%Y%m%d")


def normalize_position_backtest_positions(positions: pd.DataFrame) -> pd.DataFrame:
    assert_positions_by_rebalance_frame(positions)
    out = positions.copy()
    out["rebalance_key"] = _date_series(out["rebalance_date"]).dt.strftime("%Y%m%d")
    out["symbol"] = out["symbol"].astype(str)
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce").fillna(0.0)
    if "side" in out.columns:
        out = out.loc[out["side"].astype(str).str.lower().eq("long")].copy()
    return out.loc[out["weight"] > 0].copy()


def normalize_position_backtest_pricing(
    pricing: pd.DataFrame,
    *,
    price_col: str,
    entry_price_col: str | None = None,
    exit_price_col: str | None = None,
    tradable_col: str | None = None,
) -> pd.DataFrame:
    required = {"trade_date", "symbol", price_col}
    if entry_price_col and entry_price_col != price_col:
        required.add(entry_price_col)
    if exit_price_col and exit_price_col != price_col:
        required.add(exit_price_col)
    missing = sorted(required - set(pricing.columns))
    if missing:
        raise ValueError("Pricing file is missing required column(s): " + ", ".join(missing))
    columns = ["trade_date", "symbol", price_col]
    for col in (entry_price_col, exit_price_col):
        if col and col != price_col and col not in columns:
            columns.append(col)
    if tradable_col and tradable_col in pricing.columns:
        columns.append(tradable_col)
    out = pricing[columns].copy()
    out["trade_date"] = _date_series(out["trade_date"])
    out["symbol"] = out["symbol"].astype(str)
    out[price_col] = pd.to_numeric(out[price_col], errors="coerce")
    for col in (entry_price_col, exit_price_col):
        if col and col != price_col and col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.dropna(subset=["trade_date", "symbol", price_col]).drop_duplicates(
        subset=["trade_date", "symbol"],
        keep="last",
    )


def normalize_position_backtest_periods(periods: pd.DataFrame) -> pd.DataFrame:
    required = {"rebalance_date", "entry_date", "exit_date"}
    missing = sorted(required - set(periods.columns))
    if missing:
        raise ValueError("Periods file is missing required column(s): " + ", ".join(missing))
    out = periods.copy()
    out["rebalance_key"] = _date_series(out["rebalance_date"]).dt.strftime("%Y%m%d")
    out["entry_date_ts"] = _date_series(out["entry_date"])
    out["exit_date_ts"] = _date_series(out["exit_date"])
    out = out.dropna(subset=["rebalance_key", "entry_date_ts", "exit_date_ts"]).copy()
    out = out.sort_values(["entry_date_ts", "rebalance_key"]).reset_index(drop=True)
    if "entry_idx" not in out.columns:
        out["entry_idx"] = range(out.shape[0])
    if "exit_idx" not in out.columns:
        out["exit_idx"] = pd.to_numeric(out["entry_idx"], errors="coerce").fillna(0).astype(int) + 1
    if "planned_exit_idx" not in out.columns:
        out["planned_exit_idx"] = out["exit_idx"]
    if "planned_exit_date" not in out.columns:
        out["planned_exit_date"] = out["exit_date"]
    if "exit_delay_steps" not in out.columns:
        out["exit_delay_steps"] = (
            pd.to_numeric(out["exit_idx"], errors="coerce")
            - pd.to_numeric(out["planned_exit_idx"], errors="coerce")
        ).fillna(0)
    return out
