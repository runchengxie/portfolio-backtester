from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .benchmarking import build_benchmark_series
from .metrics import summarize_active_returns, summarize_period_returns
from .position_backtest import (
    PositionBacktestConfig,
    PositionBacktestResult,
    run_position_backtest,
)


@dataclass(frozen=True)
class PositionBacktestEvaluation:
    backtest: PositionBacktestResult
    benchmark_returns: pd.DataFrame
    active_returns: pd.DataFrame
    benchmark_stats: dict[str, Any]
    active_stats: dict[str, Any]

    @property
    def summary(self) -> dict[str, Any]:
        summary = dict(self.backtest.summary)
        summary["benchmark_stats"] = dict(self.benchmark_stats)
        summary["active_stats"] = dict(self.active_stats)
        return summary


def _strategy_return_series(result: PositionBacktestResult) -> pd.Series:
    frame = result.net_returns
    return pd.Series(
        pd.to_numeric(frame["net_return"], errors="coerce").to_numpy(dtype=float),
        index=pd.to_datetime(frame["period_end"], errors="coerce"),
        name="net_return",
    ).dropna()


def _period_records(result: PositionBacktestResult) -> list[dict[str, Any]]:
    records = result.periods.copy()
    for column in ("entry_date", "planned_exit_date", "exit_date"):
        if column in records.columns:
            records[column] = pd.to_datetime(
                records[column].astype(str),
                format="%Y%m%d",
                errors="coerce",
            )
    return records.to_dict(orient="records")


def _prepare_benchmark_frame(
    benchmark_df: pd.DataFrame | None,
    *,
    entry_price_col: str,
    exit_price_col: str,
) -> pd.DataFrame | None:
    if benchmark_df is None:
        return None
    if benchmark_df.empty:
        return benchmark_df.copy()
    required = {"trade_date", entry_price_col, exit_price_col}
    missing = sorted(required - set(benchmark_df.columns))
    if missing:
        raise ValueError("Benchmark frame is missing required column(s): " + ", ".join(missing))
    out = benchmark_df.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce").dt.normalize()
    for column in {entry_price_col, exit_price_col}:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=["trade_date", entry_price_col, exit_price_col]).copy()
    if out["trade_date"].duplicated(keep=False).any():
        raise ValueError("Benchmark frame must contain at most one row per trade_date.")
    return out.sort_values("trade_date").reset_index(drop=True)


def _return_frame(series: pd.Series, *, return_col: str) -> pd.DataFrame:
    if series is None or series.empty:
        return pd.DataFrame(columns=["period_end", return_col])
    clean = pd.to_numeric(series, errors="coerce")
    clean.index = pd.to_datetime(clean.index, errors="coerce")
    clean = clean[clean.index.notna()].dropna().sort_index()
    return pd.DataFrame(
        {
            "period_end": clean.index.strftime("%Y-%m-%d"),
            return_col: clean.to_numpy(dtype=float),
        }
    )


def evaluate_position_backtest(
    *,
    positions: pd.DataFrame,
    pricing: pd.DataFrame,
    periods: pd.DataFrame,
    config: PositionBacktestConfig,
    benchmark_return_series: pd.Series | None = None,
    benchmark_df: pd.DataFrame | None = None,
    benchmark_entry_price_col: str | None = None,
    benchmark_exit_price_col: str | None = None,
    intraday_bars: pd.DataFrame | None = None,
) -> PositionBacktestEvaluation:
    backtest = run_position_backtest(
        positions=positions,
        pricing=pricing,
        periods=periods,
        config=config,
        intraday_bars=intraday_bars,
    )
    strategy_returns = _strategy_return_series(backtest)
    period_info = _period_records(backtest)
    entry_price_col = benchmark_entry_price_col or config.effective_entry_price_col
    exit_price_col = benchmark_exit_price_col or config.effective_exit_price_col
    prepared_benchmark_df = _prepare_benchmark_frame(
        benchmark_df,
        entry_price_col=entry_price_col,
        exit_price_col=exit_price_col,
    )
    benchmark_returns, benchmark_periods = build_benchmark_series(
        prepared_benchmark_df,
        entry_price_col,
        exit_price_col,
        period_info,
        benchmark_return_series=benchmark_return_series,
    )
    periods_per_year = float(backtest.summary["stats"].get("periods_per_year", np.nan))
    benchmark_stats = summarize_period_returns(
        benchmark_returns,
        benchmark_periods,
        int(config.trading_days_per_year),
    )
    active_stats, active_returns = summarize_active_returns(
        strategy_returns,
        benchmark_returns,
        periods_per_year,
    )
    return PositionBacktestEvaluation(
        backtest=backtest,
        benchmark_returns=_return_frame(benchmark_returns, return_col="benchmark_return"),
        active_returns=_return_frame(active_returns, return_col="active_return"),
        benchmark_stats=benchmark_stats,
        active_stats=active_stats,
    )


__all__ = ["PositionBacktestEvaluation", "evaluate_position_backtest"]
