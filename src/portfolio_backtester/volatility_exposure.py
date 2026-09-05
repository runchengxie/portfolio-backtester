"""Trailing-volatility exposure decisions, to be executed after decision close."""

import numpy as np
import pandas as pd


def volatility_exposure(
    returns: pd.Series, *, window: int = 60, target_vol: float = 0.15, floor: float = 0.25
) -> pd.Series:
    """Use trailing sample standard deviation, annualized by sqrt(252).

    Warmup and zero volatility give exposure one. No leverage is permitted.
    Input must exclude synthetic initial-cash returns. The returned dates are
    decision dates, not fill dates; callers must impose execution delay.
    """
    if window < 2 or not np.isfinite(target_vol) or target_vol <= 0 or not 0 < floor <= 1:
        raise ValueError("invalid volatility-target parameters")
    if (
        not isinstance(returns.index, pd.DatetimeIndex)
        or returns.index.hasnans
        or not returns.index.is_unique
        or not returns.index.is_monotonic_increasing
        or not np.isfinite(returns).all()
    ):
        raise ValueError("returns require finite values and unique ordered datetime index")
    vol = returns.rolling(window, min_periods=window).std(ddof=1) * np.sqrt(252)
    exposure = (target_vol / vol.where(vol.gt(0))).clip(lower=floor, upper=1).fillna(1.0)
    exposure.name = "exposure"
    return exposure
