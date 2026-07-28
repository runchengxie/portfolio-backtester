"""Tearsheet statistics and return/NAV frame preparation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from ._tearsheet_render import _is_finite, _metric


def _prepare_returns(series: pd.Series | None, name: str) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float, name=name)
    work = series.copy()
    work.index = pd.to_datetime(work.index, errors="coerce")
    work = work[work.index.notna()]
    work = pd.to_numeric(work, errors="coerce").dropna().astype(float)
    work = work.sort_index()
    work.name = name
    return work


def _resolve_periods_per_year(
    *,
    strategy: pd.Series,
    strategy_stats: Mapping[str, Any] | None,
    benchmark_stats: Mapping[str, Any] | None,
) -> float:
    for stats in (strategy_stats, benchmark_stats):
        value = _metric(stats, "periods_per_year")
        if _is_finite(value) and float(value) > 0:
            return float(value)
    if strategy.shape[0] < 2:
        return np.nan
    days = float((strategy.index.max() - strategy.index.min()).days)
    if days <= 0:
        return np.nan
    return float(strategy.shape[0] / (days / 365.25))


def _summarize_series(returns: pd.Series, *, periods_per_year: float) -> dict[str, Any]:
    if returns.empty:
        return {}
    nav = (1.0 + returns).cumprod()
    total_return = float(nav.iloc[-1] - 1.0)
    max_drawdown = float((nav / nav.cummax() - 1.0).min())
    periods = int(returns.shape[0])
    if _is_finite(periods_per_year) and periods > 0:
        ann_return = float((1.0 + total_return) ** (float(periods_per_year) / periods) - 1.0)
    else:
        ann_return = np.nan
    vol = float(returns.std(ddof=1)) if periods > 1 else np.nan
    ann_vol = (
        vol * np.sqrt(periods_per_year)
        if _is_finite(vol) and _is_finite(periods_per_year)
        else np.nan
    )
    sharpe = (
        float(returns.mean() / vol * np.sqrt(periods_per_year))
        if _is_finite(vol) and vol > 0 and _is_finite(periods_per_year)
        else np.nan
    )
    downside = np.minimum(returns.to_numpy(), 0.0)
    downside_std = float(np.sqrt(np.mean(downside**2))) if downside.size else np.nan
    sortino = (
        float(returns.mean() / downside_std * np.sqrt(periods_per_year))
        if _is_finite(downside_std) and downside_std > 0 and _is_finite(periods_per_year)
        else np.nan
    )
    calmar = (
        float(ann_return / abs(max_drawdown))
        if _is_finite(ann_return) and _is_finite(max_drawdown) and max_drawdown < 0
        else np.nan
    )
    return {
        "periods": periods,
        "total_return": total_return,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "periods_per_year": periods_per_year,
        "sortino": sortino,
        "calmar": calmar,
        "skew": float(returns.skew()) if periods > 2 else np.nan,
        "kurtosis": float(returns.kurtosis()) if periods > 3 else np.nan,
        "var_95": float(np.nanpercentile(returns, 5)),
        "cvar_95": _cvar_95(returns),
        "best_period": float(returns.max()),
        "worst_period": float(returns.min()),
        "win_rate": float((returns > 0).mean()),
    }


def _merge_stats(fallback: Mapping[str, Any], stats: Mapping[str, Any] | None) -> dict[str, Any]:
    merged = dict(fallback)
    if isinstance(stats, Mapping):
        merged.update(stats)
    return merged


def _cvar_95(returns: pd.Series) -> float:
    threshold = float(np.nanpercentile(returns, 5))
    tail = returns[returns <= threshold]
    return float(tail.mean()) if not tail.empty else np.nan


def _cumulative_frame(*, strategy: pd.Series, benchmark: pd.Series) -> pd.DataFrame:
    frame = pd.DataFrame(index=strategy.index)
    frame["Strategy"] = (1.0 + strategy).cumprod() - 1.0
    if not benchmark.empty:
        benchmark_nav = (1.0 + benchmark).cumprod() - 1.0
        frame["Benchmark"] = benchmark_nav.reindex(frame.index)
    return frame.dropna(how="all")


def _layer_nav_frame(
    *,
    strategy: pd.Series,
    ideal_daily_nav_daily: pd.DataFrame | None,
    execution_sim_executed_daily: pd.DataFrame | None,
) -> pd.DataFrame:
    series: list[pd.Series] = []
    if not strategy.empty:
        series.append(((1.0 + strategy).cumprod() - 1.0).rename("Core period return"))
    ideal_nav = _daily_nav_series(ideal_daily_nav_daily, "Ideal daily NAV")
    if not ideal_nav.empty:
        series.append((ideal_nav - 1.0).rename("Ideal daily NAV"))
    executed_nav = _daily_nav_series(execution_sim_executed_daily, "Execution-adjusted NAV")
    if not executed_nav.empty:
        series.append((executed_nav - 1.0).rename("Execution-adjusted NAV"))
    if not series:
        return pd.DataFrame()
    return pd.concat(series, axis=1).sort_index().dropna(how="all")


def _daily_nav_series(daily: pd.DataFrame | None, name: str) -> pd.Series:
    if daily is None or daily.empty or "trade_date" not in daily.columns:
        return pd.Series(dtype=float, name=name)
    index = pd.to_datetime(daily["trade_date"], errors="coerce")
    if "executed_nav" in daily.columns:
        values = pd.to_numeric(daily["executed_nav"], errors="coerce")
    elif "executed_return" in daily.columns:
        returns = pd.to_numeric(daily["executed_return"], errors="coerce")
        values = (1.0 + returns).cumprod()
    else:
        return pd.Series(dtype=float, name=name)
    series = pd.Series(values.to_numpy(dtype=float), index=index, name=name)
    series = series[series.index.notna()]
    return series.dropna().sort_index()


def _rolling_sharpe_frame(
    *,
    strategy: pd.Series,
    benchmark: pd.Series,
    periods_per_year: float,
) -> pd.DataFrame:
    if strategy.empty or not _is_finite(periods_per_year) or periods_per_year <= 1:
        return pd.DataFrame()
    window = max(3, round(float(periods_per_year)))
    if strategy.shape[0] < window:
        return pd.DataFrame()
    frame = pd.DataFrame(index=strategy.index)
    frame["Strategy"] = _rolling_sharpe(strategy, window=window, periods_per_year=periods_per_year)
    if not benchmark.empty and benchmark.shape[0] >= window:
        frame["Benchmark"] = _rolling_sharpe(
            benchmark,
            window=window,
            periods_per_year=periods_per_year,
        ).reindex(frame.index)
    return frame.dropna(how="all")


def _rolling_sharpe(returns: pd.Series, *, window: int, periods_per_year: float) -> pd.Series:
    rolling_mean = returns.rolling(window, min_periods=window).mean()
    rolling_std = returns.rolling(window, min_periods=window).std(ddof=1)
    return rolling_mean / rolling_std * np.sqrt(periods_per_year)
