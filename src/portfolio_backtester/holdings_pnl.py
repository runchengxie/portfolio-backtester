"""Close-to-close marked P&L for post-trade long-only position snapshots."""

import numpy as np
import pandas as pd


def holding_pnl(units: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Return currency P&L using previous-close units and current mark changes.

    The first observation is an opening snapshot, with zero attributed P&L.
    Prices and units must use the same share basis. Missing marks are allowed
    only where no position needs valuation. No implicit price fill is performed.
    Caller-supplied suspension marks must match its ledger valuation policy.
    Trading costs and cash interest remain separate reconciliation components.
    """
    if not units.index.equals(prices.index) or not units.columns.equals(prices.columns):
        raise ValueError("units and prices must be exactly aligned")
    if (
        units.empty
        or not isinstance(units.index, pd.DatetimeIndex)
        or units.index.has_duplicates
        or units.index.hasnans
        or not units.index.is_monotonic_increasing
        or units.columns.has_duplicates
    ):
        raise ValueError("positions need a nonempty ordered unique calendar and symbols")
    if not np.isfinite(units.to_numpy()).all() or (units < 0).any().any():
        raise ValueError("units must be finite and nonnegative")
    previous = units.shift(1, fill_value=0)
    active = previous > 0
    valid = np.isfinite(prices) & (prices > 0)
    required = (units > 0) | active
    if (required & ~valid).any().any():
        raise ValueError("held positions require positive finite marks, including exit day")
    return (previous * prices.diff()).where(active, 0.0)
