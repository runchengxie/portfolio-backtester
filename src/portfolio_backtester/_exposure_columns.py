from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
import pandas as pd

from portfolio_backtester._symbol_utils import canonicalize_symbol_columns

_DEFAULT_INDUSTRY_COLUMNS = (
    "industry_name",
    "first_industry_name",
    "second_industry_name",
    "third_industry_name",
    "industry_code",
    "first_industry_code",
    "second_industry_code",
    "third_industry_code",
)

_QUALITY_COLUMNS = (
    "quality",
    "quality_score",
    "roe",
    "roe_ttm",
    "roa",
    "roa_ttm",
    "profit_margin",
    "operating_margin",
    "gross_margin",
    "gross_margin_ttm",
    "cfo_margin",
    "cfo_to_assets",
    "asset_turnover",
)

_MOMENTUM_COLUMNS = (
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

_LOW_VOL_COLUMNS = (
    "low_vol",
    "low_volatility",
    "defensive",
    "rv_120",
    "rv_60",
    "rv_20",
    "volatility_252",
    "volatility_126",
    "volatility_120",
    "volatility_60",
    "volatility_20",
)

_BETA_COLUMNS = (
    "beta",
    "beta_252",
    "beta_126",
    "beta_120",
    "beta_60",
    "market_beta",
)

_STYLE_FACTOR_ORDER = ("size", "value", "quality", "momentum", "low_vol", "beta")
_MISSING_LABEL_TOKENS = frozenset({"", "nan", "none", "<na>", "nat", "null"})


def _empty_style_summary() -> dict[str, Any]:
    return {
        "latest_rebalance_date": None,
        "latest_entry_date": None,
        "factors": {},
        "latest": {},
    }


def _empty_industry_summary() -> dict[str, Any]:
    return {
        "industry_column": None,
        "latest_rebalance_date": None,
        "latest_entry_date": None,
        "latest": {},
    }


def _exposure_period_key(value: object) -> str | None:
    return None if value is None or pd.isna(value) else str(value).strip().removesuffix(".0")


def _empty_exposure_result() -> dict[str, Any]:
    return {
        "style": pd.DataFrame(),
        "style_summary": _empty_style_summary(),
        "industry": pd.DataFrame(),
        "industry_summary": _empty_industry_summary(),
        "active_summary": pd.DataFrame(),
    }


def _to_datetime_series(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, format="%Y%m%d", errors="coerce")
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(values.loc[missing], errors="coerce")
    return parsed.dt.normalize()


def _clean_categorical_labels(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series(dtype="object", index=series.index)
    values = series.astype("string").str.strip()
    values = values.mask(values.str.lower().isin(_MISSING_LABEL_TOKENS))
    return values.astype("object")


def _as_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _zscore(series: pd.Series) -> pd.Series:
    values = _as_numeric(series)
    mask = values.notna()
    if int(mask.sum()) < 2:
        return pd.Series(np.nan, index=series.index, dtype=float)
    mean = float(values.loc[mask].mean())
    std = float(values.loc[mask].std(ddof=0))
    if not np.isfinite(std) or std <= 0:
        return pd.Series(np.nan, index=series.index, dtype=float)
    return (values - mean) / std


def _safe_log(series: pd.Series) -> pd.Series:
    values = _as_numeric(series)
    values = values.where(values > 0)
    return pd.Series(np.log(values))


def _resolve_industry_column(
    frame: pd.DataFrame,
    industry_columns: Sequence[str] | None = None,
) -> str | None:
    candidates = list(dict.fromkeys(list(industry_columns or []) + list(_DEFAULT_INDUSTRY_COLUMNS)))
    for column in candidates:
        if column not in frame.columns:
            continue
        cleaned = _clean_categorical_labels(frame[column]).dropna()
        if cleaned.empty:
            continue
        return column
    return None


def _weighted_average(values: pd.Series, weights: pd.Series) -> float:
    aligned = pd.concat(
        [values.rename("value"), pd.to_numeric(weights, errors="coerce").rename("weight")],
        axis=1,
    ).dropna()
    if aligned.empty:
        return np.nan
    total = float(aligned["weight"].sum())
    if not np.isfinite(total) or total <= 0:
        return np.nan
    return float((aligned["value"] * aligned["weight"]).sum() / total)


def _update_factor_meta(
    factor_meta: dict[str, dict[str, Any]],
    factor: str,
    meta: Mapping[str, Any],
) -> None:
    existing = factor_meta.get(factor)
    if existing is None:
        factor_meta[factor] = dict(meta)
        return
    if bool(existing.get("available")):
        return
    if bool(meta.get("available")):
        factor_meta[factor] = dict(meta)


def _price_history_tables(
    pricing_data: pd.DataFrame | None,
    *,
    price_col: str,
) -> dict[str, Any]:
    if pricing_data is None or pricing_data.empty or price_col not in pricing_data.columns:
        return {
            "price_table": pd.DataFrame(),
            "returns": pd.DataFrame(),
            "momentum_tables": {},
            "vol_tables": {},
        }
    work = pricing_data.copy()
    work = canonicalize_symbol_columns(work, context="Exposure pricing data")
    work["trade_date"] = pd.to_datetime(work["trade_date"], errors="coerce").dt.normalize()
    work = work.dropna(subset=["trade_date", "symbol"])
    work = work.drop_duplicates(subset=["trade_date", "symbol"], keep="last")
    price_table = (
        work.pivot(index="trade_date", columns="symbol", values=price_col)
        .sort_index()
        .apply(pd.to_numeric, errors="coerce")
    )
    returns = price_table.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    momentum_tables = {
        window: price_table.pct_change(window, fill_method=None).replace(
            [np.inf, -np.inf],
            np.nan,
        )
        for window in (60, 120, 252)
    }
    vol_tables = {
        window: returns.rolling(window=window, min_periods=max(20, window // 2)).std(ddof=0)
        for window in (20, 60, 120)
    }
    return {
        "price_table": price_table,
        "returns": returns,
        "momentum_tables": momentum_tables,
        "vol_tables": vol_tables,
    }


def _build_benchmark_daily_returns(
    benchmark_df: pd.DataFrame | None,
    benchmark_return_series: pd.Series | None,
    *,
    price_col: str,
) -> pd.Series:
    if benchmark_return_series is not None and not benchmark_return_series.empty:
        series = benchmark_return_series.copy()
        series.index = cast(Any, pd.to_datetime(series.index, errors="coerce")).normalize()
        series = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
        return series.dropna().sort_index()
    if benchmark_df is None or benchmark_df.empty or price_col not in benchmark_df.columns:
        return pd.Series(dtype=float, name="benchmark_return")
    work = benchmark_df.copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"], errors="coerce").dt.normalize()
    work = work.dropna(subset=["trade_date"])
    prices = (
        work.sort_values("trade_date")
        .drop_duplicates(subset=["trade_date"], keep="last")
        .set_index("trade_date")[price_col]
    )
    returns = pd.to_numeric(prices, errors="coerce").pct_change().replace([np.inf, -np.inf], np.nan)
    returns.name = "benchmark_return"
    return returns.dropna().sort_index()


def _build_beta_table(
    daily_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
) -> pd.DataFrame:
    if daily_returns.empty or benchmark_returns.empty:
        return pd.DataFrame(index=daily_returns.index, columns=daily_returns.columns, dtype=float)
    benchmark = benchmark_returns.reindex(daily_returns.index).astype(float)
    mean_stock = daily_returns.rolling(window=120, min_periods=60).mean()
    mean_bench = benchmark.rolling(window=120, min_periods=60).mean()
    mean_prod = daily_returns.mul(benchmark, axis=0).rolling(window=120, min_periods=60).mean()
    cov = mean_prod.sub(mean_stock.mul(mean_bench, axis=0), axis=0)
    var = benchmark.rolling(window=120, min_periods=60).var(ddof=0)
    beta = cov.div(var, axis=0)
    return beta.replace([np.inf, -np.inf], np.nan)
