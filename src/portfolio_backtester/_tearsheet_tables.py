"""Tearsheet HTML table builders and the final HTML assembly."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from ._tearsheet_render import (
    _MONTH_LABELS,
    _escape,
    _format_date,
    _format_metric,
    _format_number,
    _format_percent,
    _heat_class,
    _is_finite,
    _metric,
    _stylesheet,
    _year_value,
)


def _monthly_returns_table(returns: pd.Series) -> str:
    if returns.empty:
        return "<p>No data.</p>"
    monthly = returns.resample("ME").apply(lambda values: float((1.0 + values).prod() - 1.0))
    if monthly.empty:
        return "<p>No data.</p>"
    monthly_df = pd.DataFrame(
        {
            "year": monthly.index.year,
            "month": monthly.index.month,
            "return": monthly.to_numpy(dtype=float),
        }
    )
    pivot = monthly_df.pivot(index="year", columns="month", values="return").sort_index()
    rows = ['<table class="compact heatmap">']
    rows.append(
        "<thead><tr><th>Year</th>"
        + "".join(f"<th>{month}</th>" for month in _MONTH_LABELS)
        + "<th>Year</th></tr></thead>"
    )
    rows.append("<tbody>")
    for year, row in pivot.iterrows():
        year_return = float((1.0 + row.dropna()).prod() - 1.0) if row.notna().any() else np.nan
        cells = [f"<td>{int(year)}</td>"]
        for month in range(1, 13):
            value = row.get(month, np.nan)
            cells.append(
                f'<td class="{_heat_class(value)}">'
                f"{_format_percent(value, digits=2, blank='-')}</td>"
            )
        cells.append(f"<td>{_format_percent(year_return, digits=2, blank='-')}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    rows.append("</tbody></table>")
    return "\n".join(rows)


def _metrics_table(
    *,
    strategy_summary: Mapping[str, Any],
    benchmark_summary: Mapping[str, Any],
    active_stats: Mapping[str, Any] | None,
    benchmark_label: str | None,
) -> str:
    benchmark_header = benchmark_label or "Benchmark"
    rows = [
        ("Periods", "int", "periods"),
        ("Cumulative Return", "percent", "total_return"),
        ("CAGR", "percent", "ann_return"),
        ("Volatility (ann.)", "percent", "ann_vol"),
        ("Sharpe", "number", "sharpe"),
        ("Sortino", "number", "sortino"),
        ("Calmar", "number", "calmar"),
        ("Max Drawdown", "percent", "max_drawdown"),
        ("Skew", "number", "skew"),
        ("Kurtosis", "number", "kurtosis"),
        ("VaR 95%", "percent", "var_95"),
        ("Expected Shortfall 95%", "percent", "cvar_95"),
        ("Best Period", "percent", "best_period"),
        ("Worst Period", "percent", "worst_period"),
        ("Win Rate", "percent", "win_rate"),
    ]
    body = [
        "<table>",
        (
            "<thead><tr><th>Metric</th>"
            f"<th>{_escape(benchmark_header)}</th><th>Strategy</th></tr></thead>"
        ),
        "<tbody>",
    ]
    for label, kind, key in rows:
        body.append(
            "<tr>"
            f"<td>{_escape(label)}</td>"
            f"<td>{_format_metric(_metric(benchmark_summary, key), kind)}</td>"
            f"<td>{_format_metric(_metric(strategy_summary, key), kind)}</td>"
            "</tr>"
        )
    if active_stats:
        body.extend(
            [
                '<tr><td colspan="3"><hr></td></tr>',
                "<tr><td>Active Total Return</td><td>-</td><td>"
                f"{_format_metric(_metric(active_stats, 'active_total_return'), 'percent')}"
                "</td></tr>",
                "<tr><td>Tracking Error</td><td>-</td><td>"
                f"{_format_metric(_metric(active_stats, 'tracking_error'), 'percent')}</td></tr>",
                "<tr><td>Information Ratio</td><td>-</td><td>"
                f"{_format_metric(_metric(active_stats, 'information_ratio'), 'number')}</td></tr>",
                "<tr><td>Beta</td><td>-</td><td>"
                f"{_format_metric(_metric(active_stats, 'beta'), 'number')}</td></tr>",
                "<tr><td>Alpha</td><td>-</td><td>"
                f"{_format_metric(_metric(active_stats, 'alpha'), 'percent')}</td></tr>",
                "<tr><td>Correlation</td><td>-</td><td>"
                f"{_format_metric(_metric(active_stats, 'corr'), 'percent')}</td></tr>",
            ]
        )
    body.extend(["</tbody>", "</table>"])
    return "\n".join(body)


def _layer_comparison_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "<p>No data.</p>"
    rows = [
        '<table class="compact layer-table">',
        "<thead><tr>"
        "<th>Layer</th><th>Use</th><th>Status</th><th>Return</th>"
        "<th>Sharpe</th><th>Max DD</th><th>Fill</th>"
        "</tr></thead>",
        "<tbody>",
    ]
    for _, row in frame.iterrows():
        rows.append(
            "<tr>"
            f"<td>{_escape(row.get('name', '-'))}</td>"
            f"<td>{_escape(row.get('primary_use', '-'))}</td>"
            f"<td>{_escape(row.get('status', '-'))}</td>"
            f"<td>{_format_percent(row.get('total_return'), digits=2, blank='-')}</td>"
            f"<td>{_format_number(row.get('sharpe'), digits=2, blank='-')}</td>"
            f"<td>{_format_percent(row.get('max_drawdown'), digits=2, blank='-')}</td>"
            f"<td>{_format_percent(row.get('fill_ratio'), digits=2, blank='-')}</td>"
            "</tr>"
        )
    rows.extend(["</tbody>", "</table>"])
    return "\n".join(rows)


def _eoy_returns_table(
    *,
    strategy: pd.Series,
    benchmark: pd.Series,
    benchmark_label: str | None,
) -> str:
    if strategy.empty:
        return "<p>No data.</p>"
    strategy_yearly = strategy.resample("YE").apply(
        lambda values: float((1.0 + values).prod() - 1.0)
    )
    benchmark_yearly = (
        benchmark.resample("YE").apply(lambda values: float((1.0 + values).prod() - 1.0))
        if not benchmark.empty
        else pd.Series(dtype=float, index=pd.DatetimeIndex([], name="trade_date"))
    )
    years = sorted(set(strategy_yearly.index.year) | set(benchmark_yearly.index.year))
    benchmark_header = benchmark_label or "Benchmark"
    rows = [
        "<table>",
        f"<thead><tr><th>Year</th><th>{_escape(benchmark_header)}</th>"
        "<th>Strategy</th><th>Won</th></tr></thead>",
        "<tbody>",
    ]
    for year in years:
        strategy_value = _year_value(strategy_yearly, year)
        benchmark_value = _year_value(benchmark_yearly, year)
        won = (
            "+"
            if _is_finite(strategy_value)
            and _is_finite(benchmark_value)
            and strategy_value > benchmark_value
            else "-"
        )
        if not _is_finite(benchmark_value):
            won = "-"
        rows.append(
            "<tr>"
            f"<td>{year}</td>"
            f"<td>{_format_percent(benchmark_value, digits=2, blank='-')}</td>"
            f"<td>{_format_percent(strategy_value, digits=2, blank='-')}</td>"
            f"<td>{won}</td>"
            "</tr>"
        )
    rows.extend(["</tbody>", "</table>"])
    return "\n".join(rows)


def _drawdown_table(returns: pd.Series) -> str:
    periods = _drawdown_periods(returns, limit=10)
    if not periods:
        return "<p>No drawdowns.</p>"
    rows = [
        "<table>",
        "<thead><tr><th>Started</th><th>Recovered</th><th>Drawdown</th><th>Days</th></tr></thead>",
        "<tbody>",
    ]
    for period in periods:
        rows.append(
            "<tr>"
            f"<td>{_escape(period['start'])}</td>"
            f"<td>{_escape(period['recovered'])}</td>"
            f"<td>{_format_percent(period['drawdown'], digits=2, blank='-')}</td>"
            f"<td>{period['days']}</td>"
            "</tr>"
        )
    rows.extend(["</tbody>", "</table>"])
    return "\n".join(rows)


def _drawdown_periods(returns: pd.Series, *, limit: int) -> list[dict[str, Any]]:
    if returns.empty:
        return []
    nav = (1.0 + returns).cumprod()
    running_max = nav.cummax()
    drawdown = nav / running_max - 1.0
    periods: list[dict[str, Any]] = []
    in_drawdown = False
    start_date: pd.Timestamp | None = None
    trough_date: pd.Timestamp | None = None
    trough_value = 0.0
    last_date: pd.Timestamp | None = None
    for date, value in drawdown.items():
        last_date = pd.Timestamp(date)
        value = float(value)
        if value < 0 and not in_drawdown:
            in_drawdown = True
            start_date = pd.Timestamp(date)
            trough_date = pd.Timestamp(date)
            trough_value = value
        elif value < 0 and in_drawdown:
            if value < trough_value:
                trough_value = value
                trough_date = pd.Timestamp(date)
        elif value >= 0 and in_drawdown:
            recovered = pd.Timestamp(date)
            periods.append(
                _drawdown_period_record(
                    start=start_date,
                    trough=trough_date,
                    recovered=recovered,
                    drawdown=trough_value,
                )
            )
            in_drawdown = False
            start_date = None
            trough_date = None
            trough_value = 0.0
    if in_drawdown and last_date is not None:
        periods.append(
            _drawdown_period_record(
                start=start_date,
                trough=trough_date,
                recovered=None,
                fallback_end=last_date,
                drawdown=trough_value,
            )
        )
    periods.sort(key=lambda item: item["drawdown"])
    return periods[:limit]


def _drawdown_period_record(
    *,
    start: pd.Timestamp | None,
    trough: pd.Timestamp | None,
    recovered: pd.Timestamp | None,
    drawdown: float,
    fallback_end: pd.Timestamp | None = None,
) -> dict[str, Any]:
    start = start or trough or fallback_end
    end = recovered or fallback_end or trough or start
    days = int((end - start).days) if start is not None and end is not None else 0
    return {
        "start": _format_date(start),
        "trough": _format_date(trough),
        "recovered": _format_date(recovered) if recovered is not None else "-",
        "drawdown": drawdown,
        "days": days,
    }


def _render_tearsheet_html(
    *,
    title: str,
    start: str,
    end: str,
    subtitle_parts: list[str],
    charts: list[str],
    strategy: pd.Series,
    benchmark: pd.Series,
    strategy_summary: Mapping[str, Any],
    benchmark_summary: Mapping[str, Any],
    active_stats: Mapping[str, Any] | None,
    benchmark_label: str | None,
    layer_comparison: pd.DataFrame,
) -> str:
    html_parts = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '  <meta charset="utf-8">',
        '  <meta name="viewport" content="width=device-width, initial-scale=1">',
        f"  <title>{_escape(title)} Tearsheet</title>",
        "  <style>",
        _stylesheet(),
        "  </style>",
        "</head>",
        "<body>",
        '  <div class="container">',
        f"    <h1>{_escape(title)} <dt>{_escape(start)} - {_escape(end)}</dt></h1>",
        f"    <h4>{' | '.join(_escape(part) for part in subtitle_parts)}</h4>",
        "    <hr>",
        '    <div id="left">',
        *[f'      <div class="chart">{chart}</div>' for chart in charts],
        '      <div id="monthly_heatmap">',
        "        <h3>Strategy - Monthly Returns (%)</h3>",
        _monthly_returns_table(strategy),
        "      </div>",
        "    </div>",
        '    <div id="right">',
        "      <h3>Key Performance Metrics</h3>",
        _metrics_table(
            strategy_summary=strategy_summary,
            benchmark_summary=benchmark_summary,
            active_stats=active_stats,
            benchmark_label=benchmark_label,
        ),
        "      <h3>Backtest Accounting Layers</h3>",
        _layer_comparison_table(layer_comparison),
        '      <div id="eoy">',
        "        <h3>EOY Returns vs Benchmark</h3>",
        _eoy_returns_table(strategy=strategy, benchmark=benchmark, benchmark_label=benchmark_label),
        "      </div>",
        '      <div id="ddinfo">',
        "        <h3>Worst 10 Drawdowns</h3>",
        _drawdown_table(strategy),
        "      </div>",
        "    </div>",
        "  </div>",
        "</body>",
        "</html>",
    ]
    return "\n".join(html_parts)
