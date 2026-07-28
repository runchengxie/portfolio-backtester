"""Tearsheet SVG chart builders."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._tearsheet_render import (
    _escape,
    _format_axis_value,
    _tick_values,
    _x_tick_positions,
)
from ._tearsheet_stats import (
    _cumulative_frame,
    _layer_nav_frame,
    _rolling_sharpe_frame,
)


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
