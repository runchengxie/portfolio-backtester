"""Active-return (alpha/beta/IR) metrics for strategy vs benchmark."""

from __future__ import annotations

import numpy as np
import pandas as pd


def summarize_active_returns(
    strategy: pd.Series,
    benchmark: pd.Series,
    periods_per_year: float,
) -> tuple[dict[str, float], pd.Series]:
    aligned = pd.concat(
        [strategy.rename("strategy"), benchmark.rename("benchmark")], axis=1
    ).dropna()
    if aligned.empty:
        return _empty_active_summary(), pd.Series(dtype=float, name="active_return")

    strategy = aligned["strategy"]
    benchmark = aligned["benchmark"]
    active = strategy - benchmark
    mean = float(active.mean())
    std = float(active.std(ddof=1)) if active.shape[0] > 1 else np.nan
    tracking_error, information_ratio = _tracking_stats(mean, std, periods_per_year)
    beta = _beta(strategy, benchmark)
    alpha = (
        float((strategy.mean() - beta * benchmark.mean()) * periods_per_year)
        if np.isfinite(beta) and np.isfinite(periods_per_year)
        else np.nan
    )
    corr = float(strategy.corr(benchmark)) if strategy.shape[0] > 1 else np.nan
    active_total = _active_total_return(strategy, benchmark)

    return {
        "n": int(active.shape[0]),
        "mean": mean,
        "std": std,
        "tracking_error": tracking_error,
        "information_ratio": information_ratio,
        "beta": beta,
        "alpha": alpha,
        "corr": corr,
        "active_total_return": active_total,
    }, active.rename("active_return")


def _empty_active_summary() -> dict[str, float]:
    return {
        "n": 0,
        "mean": np.nan,
        "std": np.nan,
        "tracking_error": np.nan,
        "information_ratio": np.nan,
        "beta": np.nan,
        "alpha": np.nan,
        "corr": np.nan,
        "active_total_return": np.nan,
    }


def _tracking_stats(mean: float, std: float, periods_per_year: float) -> tuple[float, float]:
    if np.isfinite(std) and std > 0 and np.isfinite(periods_per_year):
        return std * np.sqrt(periods_per_year), mean / std * np.sqrt(periods_per_year)
    return np.nan, np.nan


def _beta(strategy: pd.Series, benchmark: pd.Series) -> float:
    bench_var = float(benchmark.var(ddof=1)) if benchmark.shape[0] > 1 else np.nan
    if np.isfinite(bench_var) and bench_var > 0:
        return float(strategy.cov(benchmark) / bench_var)
    return np.nan


def _active_total_return(strategy: pd.Series, benchmark: pd.Series) -> float:
    strat_total = float((1 + strategy).prod() - 1.0)
    bench_total = float((1 + benchmark).prod() - 1.0)
    if np.isfinite(strat_total) and np.isfinite(bench_total):
        active_total = (1 + strat_total) / (1 + bench_total) - 1.0
        return float(active_total) if np.isfinite(active_total) else np.nan
    return np.nan
