from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..metrics import summarize_active_returns
from ..pipeline.eval_benchmark import build_benchmark_series
from .metrics import summarize_period_returns

_ROLLING_REPORT_YEARS: tuple[int, ...] = (1, 3, 5)
_SLUG_PATTERN = re.compile(r"[^a-z0-9]+")


def slugify_report_name(name: str) -> str:
    text = str(name).strip().lower()
    if not text:
        return "benchmark"
    slug = _SLUG_PATTERN.sub("_", text).strip("_")
    return slug or "benchmark"


def build_backtest_report(
    *,
    strategy_returns: pd.Series,
    periods_per_year: float,
    benchmark_returns: pd.Series | None = None,
) -> pd.DataFrame:
    strategy = _prepare_series(strategy_returns, "strategy_return")
    if strategy.empty:
        return pd.DataFrame(
            columns=[
                "trade_date",
                "strategy_return",
                "strategy_nav",
                "benchmark_return",
                "benchmark_nav",
                "active_return",
                "relative_nav",
                *[
                    column
                    for years in _ROLLING_REPORT_YEARS
                    for column in (
                        f"strategy_rolling_cagr_{years}y",
                        f"strategy_rolling_max_drawdown_{years}y",
                    )
                ],
            ]
        )

    frame = pd.DataFrame(index=strategy.index)
    frame["strategy_return"] = strategy
    frame["strategy_nav"] = (1.0 + strategy).cumprod()

    if benchmark_returns is not None:
        benchmark = _prepare_series(benchmark_returns, "benchmark_return")
        if not benchmark.empty:
            frame["benchmark_return"] = benchmark.reindex(frame.index)
            benchmark_nav = (1.0 + benchmark).cumprod()
            frame["benchmark_nav"] = benchmark_nav.reindex(frame.index)
            frame["active_return"] = frame["strategy_return"] - frame["benchmark_return"]
            frame["relative_nav"] = frame["strategy_nav"] / frame["benchmark_nav"]
    for column in ("benchmark_return", "benchmark_nav", "active_return", "relative_nav"):
        if column not in frame.columns:
            frame[column] = np.nan

    for years in _ROLLING_REPORT_YEARS:
        cagr_col = f"strategy_rolling_cagr_{years}y"
        mdd_col = f"strategy_rolling_max_drawdown_{years}y"
        frame[cagr_col] = _rolling_cagr(strategy, periods_per_year=periods_per_year, years=years)
        frame[mdd_col] = _rolling_max_drawdown(
            strategy,
            periods_per_year=periods_per_year,
            years=years,
        )

    frame.index.name = "trade_date"
    return frame


def build_backtest_layer_comparison_frame(
    *,
    strategy_stats: Mapping[str, Any] | None,
    ideal_daily_nav_summary: Mapping[str, Any] | None = None,
    execution_sim_executed_summary: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    rows = [
        _layer_comparison_row(
            layer="core_period_return",
            name="Core period return",
            primary_use="Signal strength reference",
            summary=None,
            stats=strategy_stats,
            default_status="ok" if isinstance(strategy_stats, Mapping) else "not_run",
        ),
        _layer_comparison_row(
            layer="ideal_daily_nav",
            name="Ideal daily NAV",
            primary_use="Primary strategy NAV",
            summary=ideal_daily_nav_summary,
            stats=_nested_stats(ideal_daily_nav_summary),
            default_status="not_run",
        ),
        _layer_comparison_row(
            layer="execution_sim.executed",
            name="Execution-adjusted NAV",
            primary_use="Capacity and execution reference",
            summary=execution_sim_executed_summary,
            stats=_nested_stats(execution_sim_executed_summary),
            default_status="not_run",
        ),
    ]
    return pd.DataFrame(rows)


def build_benchmark_compare_entry(
    *,
    name: str,
    source_type: str,
    returns_file: str | None,
    symbol: str | None,
    benchmark_df: pd.DataFrame | None,
    benchmark_return_series: pd.Series | None,
    strategy_returns: pd.Series,
    period_info: list[dict[str, Any]],
    trading_days_per_year: int,
    entry_price_col: str,
    exit_price_col: str,
) -> dict[str, Any]:
    benchmark_series, benchmark_periods = build_benchmark_series(
        benchmark_df=benchmark_df,
        entry_price_col=entry_price_col,
        exit_price_col=exit_price_col,
        period_info=period_info,
        benchmark_return_series=benchmark_return_series,
    )
    benchmark_series = _prepare_series(benchmark_series, "benchmark_return")

    benchmark_stats = (
        summarize_period_returns(
            benchmark_series,
            benchmark_periods,
            trading_days_per_year,
        )
        if not benchmark_series.empty
        else None
    )
    periods_per_year = _extract_periods_per_year(benchmark_stats)
    if not np.isfinite(periods_per_year):
        periods_per_year = _infer_periods_per_year(strategy_returns, period_info)

    active_stats, active_series = summarize_active_returns(
        strategy_returns,
        benchmark_series,
        periods_per_year,
    )
    report_frame = build_backtest_report(
        strategy_returns=strategy_returns,
        periods_per_year=periods_per_year,
        benchmark_returns=benchmark_series,
    )

    return {
        "name": str(name),
        "source_type": str(source_type),
        "returns_file": str(returns_file) if returns_file else None,
        "symbol": str(symbol) if symbol else None,
        "aligned_periods": int(benchmark_series.shape[0]),
        "benchmark": benchmark_stats,
        "active": active_stats,
        "report_frame": report_frame,
        "active_series": active_series,
    }


def build_benchmark_compare_summary_frame(
    entries: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for entry in entries:
        benchmark_stats = entry.get("benchmark")
        active_stats = entry.get("active")
        rows.append(
            {
                "name": entry.get("name"),
                "source_type": entry.get("source_type"),
                "returns_file": entry.get("returns_file"),
                "symbol": entry.get("symbol"),
                "is_primary": bool(entry.get("is_primary", False)),
                "aligned_periods": entry.get("aligned_periods"),
                "benchmark_total_return": _metric_value(benchmark_stats, "total_return"),
                "benchmark_ann_return": _metric_value(benchmark_stats, "ann_return"),
                "benchmark_ann_vol": _metric_value(benchmark_stats, "ann_vol"),
                "benchmark_sharpe": _metric_value(benchmark_stats, "sharpe"),
                "benchmark_max_drawdown": _metric_value(benchmark_stats, "max_drawdown"),
                "active_tracking_error": _metric_value(active_stats, "tracking_error"),
                "active_information_ratio": _metric_value(active_stats, "information_ratio"),
                "active_beta": _metric_value(active_stats, "beta"),
                "active_alpha": _metric_value(active_stats, "alpha"),
                "active_corr": _metric_value(active_stats, "corr"),
                "active_total_return": _metric_value(active_stats, "active_total_return"),
                "report_file": entry.get("report_file"),
            }
        )
    return pd.DataFrame(rows)


def _prepare_series(series: pd.Series | None, name: str) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float, name=name)
    work = series.copy()
    work.index = pd.to_datetime(work.index, errors="coerce")
    work = work[work.index.notna()]
    work = work.sort_index()
    work = pd.to_numeric(work, errors="coerce")
    work = work[work.notna()].astype(float)
    work.name = name
    return work


def _rolling_window_obs(*, periods_per_year: float, years: int) -> int | None:
    if not np.isfinite(periods_per_year) or periods_per_year <= 0:
        return None
    window_obs = round(float(periods_per_year) * float(years))
    if window_obs <= 0:
        return None
    return window_obs


def _rolling_cagr(
    returns: pd.Series,
    *,
    periods_per_year: float,
    years: int,
) -> pd.Series:
    window_obs = _rolling_window_obs(periods_per_year=periods_per_year, years=years)
    if window_obs is None or returns.empty:
        return pd.Series(np.nan, index=returns.index, dtype=float)
    growth = (
        (1.0 + returns)
        .rolling(window_obs, min_periods=window_obs)
        .apply(
            np.prod,
            raw=True,
        )
    )
    annualization = float(periods_per_year) / float(window_obs)
    values = growth.pow(annualization) - 1.0
    return values.astype(float)


def _rolling_max_drawdown(
    returns: pd.Series,
    *,
    periods_per_year: float,
    years: int,
) -> pd.Series:
    window_obs = _rolling_window_obs(periods_per_year=periods_per_year, years=years)
    if window_obs is None or returns.empty:
        return pd.Series(np.nan, index=returns.index, dtype=float)
    return returns.rolling(window_obs, min_periods=window_obs).apply(
        _window_max_drawdown,
        raw=True,
    )


def _window_max_drawdown(values: np.ndarray) -> float:
    if values.size == 0:
        return np.nan
    nav = np.cumprod(1.0 + values)
    running_max = np.maximum.accumulate(nav)
    drawdown = nav / running_max - 1.0
    return float(np.min(drawdown))


def _extract_periods_per_year(stats: Mapping[str, Any] | None) -> float:
    if not isinstance(stats, Mapping):
        return np.nan
    value = stats.get("periods_per_year")
    try:
        periods_per_year = float(value)
    except (TypeError, ValueError):
        return np.nan
    return periods_per_year if np.isfinite(periods_per_year) and periods_per_year > 0 else np.nan


def _infer_periods_per_year(
    strategy_returns: pd.Series,
    period_info: list[dict[str, Any]],
) -> float:
    strategy = _prepare_series(strategy_returns, "strategy_return")
    if strategy.empty:
        return np.nan
    if period_info:
        holding_lengths = [
            float(info["exit_idx"] - info["entry_idx"])
            for info in period_info
            if info.get("entry_idx") is not None and info.get("exit_idx") is not None
        ]
        if holding_lengths:
            avg_holding = float(np.mean(holding_lengths))
            if np.isfinite(avg_holding) and avg_holding > 0:
                return float(252.0 / avg_holding)
    if strategy.shape[0] < 2:
        return np.nan
    start = strategy.index.min()
    end = strategy.index.max()
    total_days = float((end - start).days)
    if total_days <= 0:
        return np.nan
    return float(strategy.shape[0] / (total_days / 365.25))


def _metric_value(stats: Mapping[str, Any] | None, key: str) -> Any:
    if not isinstance(stats, Mapping):
        return None
    return stats.get(key)


def _nested_stats(summary: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if not isinstance(summary, Mapping):
        return None
    stats = summary.get("stats")
    return stats if isinstance(stats, Mapping) else None


def _layer_comparison_row(
    *,
    layer: str,
    name: str,
    primary_use: str,
    summary: Mapping[str, Any] | None,
    stats: Mapping[str, Any] | None,
    default_status: str,
) -> dict[str, Any]:
    status = (
        summary.get("status", default_status) if isinstance(summary, Mapping) else default_status
    )
    return {
        "layer": layer,
        "name": name,
        "primary_use": primary_use,
        "status": status,
        "periods": _metric_value(stats, "periods"),
        "daily_rows": _metric_value(summary, "daily_rows"),
        "periods_per_year": _metric_value(stats, "periods_per_year"),
        "total_return": _metric_value(stats, "total_return"),
        "ann_return": _metric_value(stats, "ann_return"),
        "ann_vol": _metric_value(stats, "ann_vol"),
        "sharpe": _metric_value(stats, "sharpe"),
        "max_drawdown": _metric_value(stats, "max_drawdown"),
        "fill_ratio": _metric_value(summary, "fill_ratio"),
        "buy_fill_ratio": _metric_value(summary, "buy_fill_ratio"),
        "sell_fill_ratio": _metric_value(summary, "sell_fill_ratio"),
        "unfilled_notional": _metric_value(summary, "unfilled_notional"),
        "avg_cash_weight": _metric_value(summary, "avg_cash_weight"),
        "avg_gross_exposure": _metric_value(summary, "avg_gross_exposure"),
        "final_cash_weight": _metric_value(summary, "final_cash_weight"),
        "final_gross_exposure": _metric_value(summary, "final_gross_exposure"),
        "first_trade_date": _metric_value(summary, "first_trade_date"),
        "last_trade_date": _metric_value(summary, "last_trade_date"),
    }
