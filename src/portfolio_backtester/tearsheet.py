from __future__ import annotations

import html
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .reporting import build_backtest_layer_comparison_frame

_MONTH_LABELS = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")


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


def _tearsheet_charts(
    *,
    strategy: pd.Series,
    benchmark: pd.Series,
    periods_per_year: float,
    ideal_daily_nav_daily: pd.DataFrame | None,
    execution_sim_executed_daily: pd.DataFrame | None,
) -> list[str]:
    charts = [
        _line_chart_svg(
            _cumulative_frame(strategy=strategy, benchmark=benchmark),
            title="Cumulative Returns vs Benchmark"
            if not benchmark.empty
            else "Cumulative Returns",
            value_kind="return",
        )
    ]
    layer_nav = _layer_nav_frame(
        strategy=strategy,
        ideal_daily_nav_daily=ideal_daily_nav_daily,
        execution_sim_executed_daily=execution_sim_executed_daily,
    )
    if layer_nav.shape[1] > 1:
        charts.append(
            _line_chart_svg(
                layer_nav,
                title="Backtest Layer NAV Comparison",
                value_kind="return",
            )
        )
    charts.append(_drawdown_chart_svg(strategy, title="Underwater Plot"))
    rolling = _rolling_sharpe_frame(
        strategy=strategy,
        benchmark=benchmark,
        periods_per_year=periods_per_year,
    )
    if not rolling.empty:
        charts.append(_line_chart_svg(rolling, title="Rolling Sharpe", value_kind="number"))
    return charts


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


def _line_chart_svg(frame: pd.DataFrame, *, title: str, value_kind: str) -> str:
    if frame.empty:
        return f"<h3>{_escape(title)}</h3><p>No data.</p>"
    width = 576
    height = 320
    left = 58
    right = 16
    top = 34
    bottom = 46
    values = frame.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return f"<h3>{_escape(title)}</h3><p>No data.</p>"
    y_min = float(np.nanmin(finite))
    y_max = float(np.nanmax(finite))
    if y_min == y_max:
        margin = 0.01 if y_min == 0 else abs(y_min) * 0.1
        y_min -= margin
        y_max += margin
    else:
        margin = (y_max - y_min) * 0.08
        y_min -= margin
        y_max += margin
    inner_w = width - left - right
    inner_h = height - top - bottom
    dates = list(frame.index)

    def x_at(pos: int) -> float:
        if len(dates) == 1:
            return left + inner_w / 2
        return left + inner_w * pos / (len(dates) - 1)

    def y_at(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * inner_h

    colors = ["#1f77b4", "#555555", "#2ca02c", "#9467bd"]
    polylines: list[str] = []
    legend: list[str] = []
    for idx, column in enumerate(frame.columns):
        points = [
            f"{x_at(i):.1f},{y_at(float(value)):.1f}"
            for i, value in enumerate(frame[column].to_numpy(dtype=float))
            if np.isfinite(value)
        ]
        if not points:
            continue
        color = colors[idx % len(colors)]
        polylines.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" '
            'stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        legend.append(f'<span><i style="background:{color}"></i>{_escape(str(column))}</span>')

    y_ticks = _tick_values(y_min, y_max, 5)
    y_axis = "\n".join(
        (
            f'<line x1="{left}" x2="{width - right}" y1="{y_at(value):.1f}" '
            f'y2="{y_at(value):.1f}" class="grid"/>'
            f'<text x="{left - 8}" y="{y_at(value) + 4:.1f}" class="axis" text-anchor="end">'
            f"{_escape(_format_axis_value(value, value_kind=value_kind))}</text>"
        )
        for value in y_ticks
    )
    x_axis = "\n".join(
        f'<text x="{x_at(pos):.1f}" y="{height - 16}" class="axis" text-anchor="middle">'
        f"{_escape(pd.Timestamp(dates[pos]).strftime('%Y-%m'))}</text>"
        for pos in _x_tick_positions(len(dates))
    )
    return (
        f"<h3>{_escape(title)}</h3>"
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{_escape(title)}">'
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#fff"/>'
        f'<line x1="{left}" x2="{left}" y1="{top}" y2="{height - bottom}" class="axis-line"/>'
        f'<line x1="{left}" x2="{width - right}" y1="{height - bottom}" '
        f'y2="{height - bottom}" class="axis-line"/>'
        f"{y_axis}{x_axis}{''.join(polylines)}"
        f'</svg><div class="legend">{"".join(legend)}</div>'
    )


def _drawdown_chart_svg(returns: pd.Series, *, title: str) -> str:
    if returns.empty:
        return f"<h3>{_escape(title)}</h3><p>No data.</p>"
    nav = (1.0 + returns).cumprod()
    drawdown = nav / nav.cummax() - 1.0
    frame = pd.DataFrame({"Drawdown": drawdown})
    width = 576
    height = 240
    left = 58
    right = 16
    top = 28
    bottom = 42
    y_min = min(float(drawdown.min()), -0.01)
    y_max = 0.0
    inner_w = width - left - right
    inner_h = height - top - bottom
    dates = list(frame.index)

    def x_at(pos: int) -> float:
        if len(dates) == 1:
            return left + inner_w / 2
        return left + inner_w * pos / (len(dates) - 1)

    def y_at(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * inner_h

    points = [f"{x_at(i):.1f},{y_at(float(value)):.1f}" for i, value in enumerate(drawdown)]
    baseline = y_at(0.0)
    polygon = (
        f"{left:.1f},{baseline:.1f} "
        + " ".join(points)
        + f" {x_at(len(dates) - 1):.1f},{baseline:.1f}"
    )
    y_ticks = _tick_values(y_min, y_max, 4)
    y_axis = "\n".join(
        (
            f'<line x1="{left}" x2="{width - right}" y1="{y_at(value):.1f}" '
            f'y2="{y_at(value):.1f}" class="grid"/>'
            f'<text x="{left - 8}" y="{y_at(value) + 4:.1f}" class="axis" text-anchor="end">'
            f"{_escape(_format_axis_value(value, value_kind='return'))}</text>"
        )
        for value in y_ticks
    )
    x_axis = "\n".join(
        f'<text x="{x_at(pos):.1f}" y="{height - 14}" class="axis" text-anchor="middle">'
        f"{_escape(pd.Timestamp(dates[pos]).strftime('%Y-%m'))}</text>"
        for pos in _x_tick_positions(len(dates))
    )
    return (
        f"<h3>{_escape(title)}</h3>"
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{_escape(title)}">'
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#fff"/>'
        f"{y_axis}{x_axis}"
        f'<polygon points="{polygon}" fill="#f3b7b7" opacity="0.7"/>'
        f'<polyline points="{" ".join(points)}" fill="none" stroke="#b23a3a" stroke-width="1.8"/>'
        f'<line x1="{left}" x2="{width - right}" y1="{baseline:.1f}" y2="{baseline:.1f}" '
        'class="axis-line"/>'
        "</svg>"
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


def _year_value(series: pd.Series, year: int) -> float:
    if series.empty:
        return np.nan
    matches = series[series.index.year == year]
    return float(matches.iloc[0]) if not matches.empty else np.nan


def _date_range(series: pd.Series) -> tuple[str, str]:
    if series.empty:
        return "-", "-"
    return _format_date(series.index.min()), _format_date(series.index.max())


def _format_date(value: Any) -> str:
    if value is None or pd.isna(value):
        return "-"
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _format_metric(value: Any, kind: str) -> str:
    if kind == "percent":
        return _format_percent(value, digits=2, blank="-")
    if kind == "int":
        if not _is_finite(value):
            return "-"
        return f"{int(value):,}"
    return _format_number(value, digits=2, blank="-")


def _format_percent(value: Any, *, digits: int, blank: str) -> str:
    if not _is_finite(value):
        return blank
    return f"{float(value) * 100.0:,.{digits}f}%"


def _format_number(value: Any, *, digits: int, blank: str = "-") -> str:
    if not _is_finite(value):
        return blank
    return f"{float(value):,.{digits}f}"


def _format_axis_value(value: float, *, value_kind: str) -> str:
    if value_kind == "return":
        return _format_percent(value, digits=0, blank="-")
    return _format_number(value, digits=1, blank="-")


def _metric(stats: Mapping[str, Any] | None, key: str) -> Any:
    if not isinstance(stats, Mapping):
        return np.nan
    return stats.get(key, np.nan)


def _is_finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _tick_values(min_value: float, max_value: float, count: int) -> list[float]:
    if count <= 1 or min_value == max_value:
        return [min_value, max_value]
    return [float(value) for value in np.linspace(min_value, max_value, count)]


def _x_tick_positions(length: int) -> list[int]:
    if length <= 1:
        return [0]
    if length <= 4:
        return list(range(length))
    raw = [0, length // 4, length // 2, (length * 3) // 4, length - 1]
    return sorted(set(raw))


def _heat_class(value: Any) -> str:
    if not _is_finite(value):
        return "heat-empty"
    value = float(value)
    if value > 0.05:
        return "heat-pos-strong"
    if value > 0:
        return "heat-pos"
    if value < -0.05:
        return "heat-neg-strong"
    if value < 0:
        return "heat-neg"
    return "heat-flat"


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _stylesheet() -> str:
    return """
body{-webkit-font-smoothing:antialiased;-moz-osx-font-smoothing:grayscale;margin:30px;background:#fff;color:#111}
body,p,table,td,th{font:13px/1.4 Arial,sans-serif}
.container{max-width:980px;margin:auto}
h1,h2,h3,h4{font-weight:400;margin:0}
h1 dt{display:inline;margin-left:10px;font-size:14px;color:#555}
h3{margin:0 0 10px;font-weight:700}
h4{color:#666;margin-top:4px}
hr{margin:25px 0 34px;height:0;border:0;border-top:1px solid #ccc}
#left{width:620px;margin-right:22px;float:left}
#right{width:330px;float:right}
.chart{margin:0 0 28px}
svg{width:100%;height:auto}
.axis{fill:#666;font-size:11px}
.axis-line{stroke:#999;stroke-width:0.8}
.grid{stroke:#ddd;stroke-width:0.7}
.legend{margin:-4px 0 12px;color:#555}
.legend span{display:inline-block;margin-right:14px}
.legend i{display:inline-block;width:18px;height:3px;margin-right:5px;vertical-align:middle}
table{margin:0 0 32px;border:0;border-spacing:0;width:100%}
table td,table th{text-align:right;padding:4px 5px 3px}
table th{padding:6px 5px 5px;font-weight:700;background:#eee}
table td:first-of-type,table th:first-of-type{text-align:left;padding-left:2px}
table td:last-of-type,table th:last-of-type{padding-right:2px}
td hr{margin:5px 0}
.compact td,.compact th{font-size:12px;padding:3px 4px;text-align:right}
.layer-table td,.layer-table th{font-size:11px}
.heat-empty{color:#999;background:#f7f7f7}
.heat-pos{background:#e7f3ea}
.heat-pos-strong{background:#b8dfc1}
.heat-neg{background:#fae6e6}
.heat-neg-strong{background:#efb8b8}
.heat-flat{background:#f4f4f4}
@media (max-width: 900px){body{margin:18px}#left,#right{float:none;width:100%;margin:0}}
@media (max-width: 900px){h1 dt{display:block;margin-left:0;margin-top:4px}}
@media print{body{margin:0}.container{max-width:100%;margin:0}}
@media print{#left{width:58%;margin:0 2% 0 0}#right{width:40%}hr{margin:20px 0}}
""".strip()
