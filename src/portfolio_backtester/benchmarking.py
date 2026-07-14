from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


def warn_if_delay_exit_lag(
    *,
    label_prefix: str,
    exit_price_policy: str,
    stats: Mapping[str, Any] | None,
) -> None:
    logger = logging.getLogger("portfolio_backtester")
    if str(exit_price_policy).strip().lower() != "delay":
        return
    if not isinstance(stats, Mapping):
        return
    delayed_raw = stats.get("periods_with_delayed_exit")
    periods_raw = stats.get("periods")
    try:
        delayed_periods = int(delayed_raw) if delayed_raw is not None else 0
    except (TypeError, ValueError):
        delayed_periods = 0
    try:
        total_periods = int(periods_raw) if periods_raw is not None else 0
    except (TypeError, ValueError):
        total_periods = 0
    if delayed_periods <= 0:
        return
    avg_lag = stats.get("avg_exit_lag_days")
    max_lag = stats.get("max_exit_lag_days")
    try:
        avg_value = float(avg_lag)
    except (TypeError, ValueError):
        avg_value = np.nan
    try:
        max_value = float(max_lag)
    except (TypeError, ValueError):
        max_value = np.nan
    avg_text = f"{avg_value:.2f}" if np.isfinite(avg_value) else "nan"
    max_text = f"{max_value:.0f}" if np.isfinite(max_value) else "nan"
    logger.warning(
        "%sDelay exit policy produced lagged exits in %s/%s periods "
        "(avg_lag=%s, max_lag=%s trade days).",
        label_prefix,
        delayed_periods,
        total_periods,
        avg_text,
        max_text,
    )


def build_benchmark_series(
    benchmark_df: pd.DataFrame | None,
    entry_price_col: str,
    exit_price_col: str,
    period_info: list[dict[str, Any]],
    benchmark_return_series: pd.Series | None = None,
) -> tuple[pd.Series, list[dict[str, Any]]]:
    if benchmark_return_series is not None and not benchmark_return_series.empty:
        aligned_returns = benchmark_return_series.copy()
        aligned_returns.index = pd.to_datetime(aligned_returns.index, errors="coerce")
        aligned_returns = aligned_returns[aligned_returns.index.notna()]
        aligned_returns = aligned_returns.sort_index()
        aligned_returns = pd.to_numeric(aligned_returns, errors="coerce").dropna()

        bench_returns = []
        bench_index = []
        bench_periods: list[dict[str, Any]] = []
        for info in period_info:
            entry_date = pd.to_datetime(info.get("entry_date"), errors="coerce")
            exit_date = pd.to_datetime(info["exit_date"], errors="coerce")
            if pd.isna(entry_date) or pd.isna(exit_date):
                continue

            window = aligned_returns[
                (aligned_returns.index > entry_date) & (aligned_returns.index <= exit_date)
            ]
            if window.empty and exit_date in aligned_returns.index:
                value = aligned_returns.loc[exit_date]
                window = value if isinstance(value, pd.Series) else pd.Series([value])
            if window.empty:
                continue
            bench_value = float((1.0 + window.astype(float)).prod() - 1.0)
            if not np.isfinite(bench_value):
                continue
            bench_returns.append(bench_value)
            bench_index.append(exit_date)
            bench_periods.append(info)
        if bench_returns:
            return pd.Series(
                bench_returns,
                index=bench_index,
                name="benchmark_return",
            ), bench_periods
    if benchmark_df is None or benchmark_df.empty:
        return pd.Series(dtype=float, name="benchmark_return"), []
    if entry_price_col not in benchmark_df.columns or exit_price_col not in benchmark_df.columns:
        return pd.Series(dtype=float, name="benchmark_return"), []
    bench_entry_prices = benchmark_df.set_index("trade_date")[entry_price_col]
    bench_exit_prices = benchmark_df.set_index("trade_date")[exit_price_col]
    bench_returns = []
    bench_index = []
    bench_periods: list[dict[str, Any]] = []
    for info in period_info:
        entry_date = info["entry_date"]
        exit_date = info["exit_date"]
        if entry_date not in bench_entry_prices.index or exit_date not in bench_exit_prices.index:
            continue
        bench_returns.append(
            bench_exit_prices.loc[exit_date] / bench_entry_prices.loc[entry_date] - 1.0
        )
        bench_index.append(exit_date)
        bench_periods.append(info)
    if not bench_returns:
        return pd.Series(dtype=float, name="benchmark_return"), []
    return pd.Series(bench_returns, index=bench_index, name="benchmark_return"), bench_periods


__all__ = ["build_benchmark_series", "warn_if_delay_exit_lag"]
