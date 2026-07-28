"""Backtest tearsheet (HTML) generation.

The heavy lifting is split into private submodules:

* :mod:`portfolio_backtester._tearsheet_render` — escaping/formatting/stylesheet
* :mod:`portfolio_backtester._tearsheet_stats` — stats and NAV/return frames
* :mod:`portfolio_backtester._tearsheet_charts` — SVG charts
* :mod:`portfolio_backtester._tearsheet_tables` — HTML tables and final assembly

The two public callables ``write_backtest_tearsheet`` and
``build_backtest_tearsheet_html`` remain defined on this shell module so that
``from .tearsheet import *`` (used by :mod:`portfolio_backtester.reports`) keeps
exporting exactly the same names as before.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ._tearsheet_charts import _tearsheet_charts
from ._tearsheet_render import _date_range, _format_number
from ._tearsheet_stats import (
    _merge_stats,
    _prepare_returns,
    _resolve_periods_per_year,
    _summarize_series,
)
from ._tearsheet_tables import _render_tearsheet_html
from .reporting import build_backtest_layer_comparison_frame


def write_backtest_tearsheet(
    *,
    path: Path,
    strategy_returns: pd.Series,
    strategy_stats: Mapping[str, Any] | None,
    benchmark_returns: pd.Series | None,
    benchmark_stats: Mapping[str, Any] | None,
    active_stats: Mapping[str, Any] | None,
    title: str,
    benchmark_name: str | None = None,
    generated_at: datetime | None = None,
    ideal_daily_nav_summary: Mapping[str, Any] | None = None,
    ideal_daily_nav_daily: pd.DataFrame | None = None,
    execution_sim_executed_summary: Mapping[str, Any] | None = None,
    execution_sim_executed_daily: pd.DataFrame | None = None,
) -> None:
    content = build_backtest_tearsheet_html(
        strategy_returns=strategy_returns,
        strategy_stats=strategy_stats,
        benchmark_returns=benchmark_returns,
        benchmark_stats=benchmark_stats,
        active_stats=active_stats,
        title=title,
        benchmark_name=benchmark_name,
        generated_at=generated_at,
        ideal_daily_nav_summary=ideal_daily_nav_summary,
        ideal_daily_nav_daily=ideal_daily_nav_daily,
        execution_sim_executed_summary=execution_sim_executed_summary,
        execution_sim_executed_daily=execution_sim_executed_daily,
    )
    path.write_text(content, encoding="utf-8")


def build_backtest_tearsheet_html(
    *,
    strategy_returns: pd.Series,
    strategy_stats: Mapping[str, Any] | None = None,
    benchmark_returns: pd.Series | None = None,
    benchmark_stats: Mapping[str, Any] | None = None,
    active_stats: Mapping[str, Any] | None = None,
    title: str = "Backtest Tearsheet",
    benchmark_name: str | None = None,
    generated_at: datetime | None = None,
    ideal_daily_nav_summary: Mapping[str, Any] | None = None,
    ideal_daily_nav_daily: pd.DataFrame | None = None,
    execution_sim_executed_summary: Mapping[str, Any] | None = None,
    execution_sim_executed_daily: pd.DataFrame | None = None,
) -> str:
    strategy = _prepare_returns(strategy_returns, "strategy")
    benchmark = _prepare_returns(benchmark_returns, "benchmark")
    benchmark = benchmark.reindex(strategy.index).dropna() if not benchmark.empty else benchmark
    periods_per_year = _resolve_periods_per_year(
        strategy=strategy,
        strategy_stats=strategy_stats,
        benchmark_stats=benchmark_stats,
    )
    strategy_summary = _merge_stats(
        _summarize_series(strategy, periods_per_year=periods_per_year),
        strategy_stats,
    )
    benchmark_summary = (
        _merge_stats(
            _summarize_series(benchmark, periods_per_year=periods_per_year),
            benchmark_stats,
        )
        if not benchmark.empty
        else {}
    )
    start, end = _date_range(strategy)
    generated_at = generated_at or datetime.now()
    benchmark_label = benchmark_name or ("Benchmark" if not benchmark.empty else None)
    layer_comparison = build_backtest_layer_comparison_frame(
        strategy_stats=strategy_summary,
        ideal_daily_nav_summary=ideal_daily_nav_summary,
        execution_sim_executed_summary=execution_sim_executed_summary,
    )
    charts = _tearsheet_charts(
        strategy=strategy,
        benchmark=benchmark,
        periods_per_year=periods_per_year,
        ideal_daily_nav_daily=ideal_daily_nav_daily,
        execution_sim_executed_daily=execution_sim_executed_daily,
    )
    return _render_tearsheet_html(
        title=title,
        start=start,
        end=end,
        subtitle_parts=_subtitle_parts(
            periods_per_year=periods_per_year,
            generated_at=generated_at,
            benchmark_label=benchmark_label,
        ),
        charts=charts,
        strategy=strategy,
        benchmark=benchmark,
        strategy_summary=strategy_summary,
        benchmark_summary=benchmark_summary,
        active_stats=active_stats,
        benchmark_label=benchmark_label,
        layer_comparison=layer_comparison,
    )


def _subtitle_parts(
    *,
    periods_per_year: float,
    generated_at: datetime,
    benchmark_label: str | None,
) -> list[str]:
    parts = [
        f"Periods/Year: {_format_number(periods_per_year, digits=2)}",
        f"Generated: {generated_at.strftime('%Y-%m-%d %H:%M:%S')}",
    ]
    if benchmark_label:
        parts.insert(0, f"Benchmark: {benchmark_label}")
    return parts
