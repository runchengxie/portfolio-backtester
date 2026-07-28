"""Portfolio performance metrics.

This module re-exports the public metric helpers, which now live in the
private submodules (:mod:`_metrics_ic`, :mod:`_metrics_active`,
:mod:`_metrics_turnover`, :mod:`_metrics_period`). The original behavior is
unchanged; symbols remain importable from this path.
"""

from __future__ import annotations

from ._metrics_active import summarize_active_returns as summarize_active_returns
from ._metrics_ic import (
    daily_ic_series as daily_ic_series,
    quantile_returns as quantile_returns,
    summarize_ic as summarize_ic,
)
from ._metrics_period import summarize_period_returns as summarize_period_returns
from ._metrics_turnover import estimate_turnover as estimate_turnover

__all__ = [
    "daily_ic_series",
    "estimate_turnover",
    "quantile_returns",
    "summarize_active_returns",
    "summarize_ic",
    "summarize_period_returns",
]
