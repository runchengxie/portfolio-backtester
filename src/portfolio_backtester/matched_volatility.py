"""Ex-post realized-volatility matching, never an executable allocation rule."""

import numpy as np
import pandas as pd


def match_realized_volatility(returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Scale each column to the smallest full-sample realized volatility.

    Uses future observations by design. Scaling net returns is an analytical
    approximation: it does not recompute turnover, fees, fills or cash yield.
    No column is levered up. Inputs must already have identical observation dates.
    """
    if (
        not isinstance(returns.index, pd.DatetimeIndex)
        or returns.index.hasnans
        or not returns.index.is_unique
        or not returns.index.is_monotonic_increasing
        or len(returns) < 2
        or returns.shape[1] == 0
        or not returns.columns.is_unique
        or not np.isfinite(returns).all().all()
    ):
        raise ValueError("finite aligned return columns and unique ordered dates required")
    vol = returns.std(ddof=1)
    if not vol.gt(0).all():
        raise ValueError("strictly positive realized volatility required")
    scales = vol.min() / vol
    return returns.mul(scales), scales
