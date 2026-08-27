"""Portfolio reporting metrics and compatibility helpers.

Portfolio-owned public semantics are turnover, active-return, period-return and
leg-attribution measures. The IC/quantile helpers re-exported from this legacy
module remain for package compatibility and attribution internals; canonical
alpha-research IC, Pearson IC, IC-IR and quantile research semantics belong to
``alpha-research``. New portfolio APIs should not treat these helpers as an
alpha-evaluation owner surface.
"""

from __future__ import annotations

import pandas as pd

from . import _metrics_ic
from ._metrics_active import summarize_active_returns as summarize_active_returns
from ._metrics_ic import (
    daily_ic_series as daily_ic_series,
    leg_attribution_frame as leg_attribution_frame,
    pearson_corr as pearson_corr,
    quantile_returns as quantile_returns,
    spearman_corr as spearman_corr,
    summarize_leg_attribution as summarize_leg_attribution,
)
from ._metrics_period import summarize_period_returns as summarize_period_returns
from ._metrics_turnover import estimate_turnover as estimate_turnover

# Keep the legacy monkeypatch seam used by downstream optional-dependency tests.
scipy_stats = _metrics_ic.scipy_stats


def summarize_ic(ic_series: pd.Series) -> dict[str, float]:
    """Summarize compatibility IC while honoring the SciPy override."""
    _metrics_ic.scipy_stats = scipy_stats
    return _metrics_ic.summarize_ic(ic_series)


__all__ = [
    "daily_ic_series",
    "estimate_turnover",
    "leg_attribution_frame",
    "pearson_corr",
    "quantile_returns",
    "spearman_corr",
    "summarize_active_returns",
    "summarize_ic",
    "summarize_leg_attribution",
    "summarize_period_returns",
]
