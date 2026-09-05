"""Drawdown episodes retaining right-censored recoveries at the observation end."""

import numpy as np
import pandas as pd

_COLUMNS = [
    "peak_date",
    "trough_date",
    "recovery_date",
    "observation_end",
    "depth",
    "underwater_sessions",
    "elapsed_calendar_days",
    "censored",
]


def drawdown_episodes(nav: pd.Series) -> pd.DataFrame:
    """Measure each excursion below the last high-water mark.

    An observation at or above the previous peak closes an episode. Equal peaks
    refresh the starting date. Underwater sessions count strictly-below-peak
    observations, excluding the recovery observation. Elapsed days run from the
    peak through recovery, or through the last observation for censored episodes.
    No dates or missing NAVs are silently removed. Input observations should be
    trading-session closes when interpreting the session-count field.
    """
    if (
        not isinstance(nav.index, pd.DatetimeIndex)
        or nav.index.hasnans
        or not nav.index.is_unique
        or not nav.index.is_monotonic_increasing
    ):
        raise ValueError("NAV requires unique, ordered nonmissing datetime observations")
    values = nav.to_numpy(dtype=float)
    if not len(values) or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("NAV must be nonempty, finite and positive")
    rows = []
    peak = trough = 0
    underwater = 0
    for i in range(1, len(values)):
        recovered = values[i] >= values[peak]
        if not recovered:
            underwater += 1
            if values[i] < values[trough]:
                trough = i
        if underwater and (recovered or i == len(values) - 1):
            rows.append(
                {
                    "peak_date": nav.index[peak],
                    "trough_date": nav.index[trough],
                    "recovery_date": nav.index[i] if recovered else pd.NaT,
                    "observation_end": nav.index[i],
                    "depth": values[trough] / values[peak] - 1,
                    "underwater_sessions": underwater,
                    "elapsed_calendar_days": (nav.index[i] - nav.index[peak]).days,
                    "censored": not recovered,
                }
            )
        if recovered:
            peak = trough = i
            underwater = 0
    return pd.DataFrame(rows, columns=_COLUMNS)
