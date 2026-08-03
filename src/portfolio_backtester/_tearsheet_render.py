"""Tearsheet rendering helpers: escaping, number/date formatting, stylesheets."""

from __future__ import annotations

import html
from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import pandas as pd

_MONTH_LABELS = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


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


def _format_metric(value: Any, kind: str) -> str:
    if kind == "percent":
        return _format_percent(value, digits=2, blank="-")
    if kind == "int":
        if not _is_finite(value):
            return "-"
        return f"{int(value):,}"
    return _format_number(value, digits=2, blank="-")


def _metric(stats: Mapping[str, Any] | None, key: str) -> Any:
    if not isinstance(stats, Mapping):
        return np.nan
    return stats.get(key, np.nan)


def _is_finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _format_date(value: Any) -> str:
    if value is None or pd.isna(value):
        return "-"
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _date_range(series: pd.Series) -> tuple[str, str]:
    if series.empty:
        return "-", "-"
    return _format_date(series.index.min()), _format_date(series.index.max())


def _year_value(series: pd.Series, year: int) -> float:
    if series.empty:
        return np.nan
    matches = series[cast(Any, series.index).year == year]
    return float(matches.iloc[0]) if not matches.empty else np.nan


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
