"""Post-buffer exposure repair: configuration, momentum z-score, position helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

MOMENTUM_COLUMNS = (
    "momentum",
    "momentum_12m",
    "momentum_6m",
    "mom_12m",
    "mom_6m",
    "ret_252",
    "ret_126",
    "ret_120",
    "ret_60",
    "ret_20",
    "ret_5",
)


@dataclass(frozen=True)
class PostBufferExposureRepairConfig:
    strict_guardrail_min_rank: float = 0.70
    bank_fallback_min_rank: float | None = 0.65
    bank_fallback_min_signal: float = 0.0
    exposure_margin: float = 0.003
    bank_industry_name: str = "银行"
    industry_col: str = "first_industry_name"
    signal_col: str = "signal_z"
    guardrail_col: str = "earnings_burst_rank"
    momentum_col: str = "exposure_momentum_z"
    tradable_col: str = "is_tradable"
    max_abs_industry_active: float = 0.20
    max_abs_momentum_active: float = 1.0


@dataclass(frozen=True)
class PostBufferExposureRepairResult:
    positions: pd.DataFrame
    actions: list[dict[str, Any]]


def add_exposure_momentum_z(
    source: pd.DataFrame,
    *,
    momentum_col: str = "exposure_momentum_z",
) -> pd.DataFrame:
    """Add the same momentum z-score used by exposure analysis when source columns exist."""
    work = source.copy()
    if momentum_col in work.columns:
        work["date_i"] = _compact_date_series(work["trade_date"]).astype(int)
        return work
    columns = [column for column in MOMENTUM_COLUMNS if column in work.columns]
    if not columns:
        work[momentum_col] = np.nan
    else:
        components = [pd.to_numeric(work[column], errors="coerce") for column in columns]
        work["_exposure_momentum_raw"] = pd.concat(components, axis=1).mean(axis=1, skipna=True)
        work[momentum_col] = work.groupby("trade_date")["_exposure_momentum_raw"].transform(_zscore)
        work.drop(columns=["_exposure_momentum_raw"], inplace=True)
    work["date_i"] = _compact_date_series(work["trade_date"]).astype(int)
    return work


def _prepare_repair_source(
    source: pd.DataFrame,
    config: PostBufferExposureRepairConfig,
) -> pd.DataFrame:
    work = source.copy()
    if config.signal_col not in work.columns:
        fallback_signal_col = next(
            (
                column
                for column in ("signal_backtest", "signal_eval", "pred", "signal_z", "signal")
                if column in work.columns
            ),
            None,
        )
        work[config.signal_col] = (
            pd.to_numeric(work[fallback_signal_col], errors="coerce")
            if fallback_signal_col is not None
            else np.nan
        )
    if config.guardrail_col not in work.columns:
        signal = pd.to_numeric(work[config.signal_col], errors="coerce")
        if "trade_date" in work.columns:
            work[config.guardrail_col] = signal.groupby(work["trade_date"]).rank(pct=True)
        else:
            work[config.guardrail_col] = signal.rank(pct=True)
    if config.industry_col not in work.columns:
        work[config.industry_col] = ""
    if config.tradable_col not in work.columns:
        work[config.tradable_col] = True
    return work


def normalize_repair_positions(positions: pd.DataFrame) -> pd.DataFrame:
    out = positions.copy()
    for column in ("rebalance_date", "entry_date"):
        if column not in out.columns:
            raise ValueError(f"positions must include {column}.")
        out[column] = _compact_date_series(out[column]).astype(int)
    if "weight" not in out.columns:
        raise ValueError("positions must include weight.")
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce").fillna(0.0)
    if "signal" not in out.columns:
        out["signal"] = 0.0
    if "rank" not in out.columns:
        out["rank"] = 0
    if "side" not in out.columns:
        out["side"] = "long"
    out = out.loc[out["weight"] > 1e-10].copy()
    grouped = (
        out.groupby(["rebalance_date", "symbol"], as_index=False)
        .agg(
            entry_date=("entry_date", "first"),
            weight=("weight", "sum"),
            signal=("signal", "max"),
            rank=("rank", "min"),
            side=("side", "first"),
        )
        .copy()
    )
    for _, idx in grouped.groupby("rebalance_date").groups.items():
        total = float(grouped.loc[idx, "weight"].sum())
        if total > 0:
            grouped.loc[idx, "weight"] = grouped.loc[idx, "weight"] / total
        signals = pd.to_numeric(grouped.loc[idx, "signal"], errors="coerce").fillna(-1e9)
        grouped.loc[idx, "rank"] = signals.rank(method="first", ascending=False).astype(int)
    grouped["side"] = "long"
    return grouped.sort_values(["rebalance_date", "rank", "symbol"]).reset_index(drop=True)[
        ["rebalance_date", "entry_date", "symbol", "weight", "signal", "rank", "side"]
    ]


def _compact_date_series(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    compact_mask = text.str.fullmatch(r"\d{8}")
    parsed = pd.to_datetime(text, errors="coerce")
    compact = parsed.dt.strftime("%Y%m%d").mask(compact_mask, text)
    return pd.to_numeric(compact, errors="coerce").astype("Int64")


def _zscore(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    std = numeric.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return pd.Series(np.nan, index=values.index)
    return (numeric - numeric.mean()) / std


def _breach_rows(breaches: pd.DataFrame) -> pd.DataFrame:
    if breaches is None or breaches.empty:
        return pd.DataFrame()
    work = breaches.copy()
    if "status" in work.columns:
        work = work.loc[work["status"].astype(str).str.lower() == "breached"]
    if "rebalance_date" in work.columns:
        work["rebalance_date"] = _compact_date_series(work["rebalance_date"]).astype(int)
    return work


def _is_momentum_breach(row: pd.Series) -> bool:
    return str(row.get("check")) == "style_active" and str(row.get("name")) == "momentum"


def _is_bank_breach(row: pd.Series, config: PostBufferExposureRepairConfig) -> bool:
    return (
        str(row.get("check")) == "industry_active"
        and str(row.get("name")) == config.bank_industry_name
    )
