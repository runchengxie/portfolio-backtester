"""Fractional-share next-close diagnostic replay on adjusted price units.

Held or traded prices must be complete on the caller's explicit calendar, except
explicit suspension marking. Caller-supplied directional blocks defer trades.
Lot rounding, exchange queue fills and a separate dividend ledger are not modeled.
"""

import numpy as np
import pandas as pd


def _execution_schedule(targets: pd.DataFrame, prices: pd.DataFrame) -> dict:
    """Validate inputs and map formations to strictly later execution closes."""
    if prices.empty or not prices.index.is_monotonic_increasing or prices.index.has_duplicates:
        raise ValueError("prices need a nonempty ordered unique calendar")
    if prices.columns.has_duplicates:
        raise ValueError("duplicate price symbols")
    work = targets.copy()
    work["formation_date"] = pd.to_datetime(work["formation_date"])
    if work[["formation_date", "symbol"]].isna().any().any():
        raise ValueError("target keys must be present")
    if work.duplicated(["formation_date", "symbol"]).any():
        raise ValueError("duplicate target symbols")
    if not set(work.symbol).issubset(prices.columns):
        raise ValueError("missing target prices")
    if not np.isfinite(work.weight).all() or (work.weight < 0).any():
        raise ValueError("invalid target weights")
    schedule = {}
    for formation, group in work.groupby("formation_date", sort=True):
        if not isinstance(formation, pd.Timestamp):
            raise ValueError("formation dates must be timestamps")
        if group.weight.sum() > 1 + 1e-12:
            raise ValueError("target weights exceed capital")
        idx = int(np.searchsorted(prices.index.to_numpy(), formation.to_datetime64(), side="right"))
        if idx == len(prices):
            continue
        if idx in schedule:
            raise ValueError("multiple formations map to one execution close")
        schedule[idx] = (
            group.set_index("symbol").weight.reindex(prices.columns, fill_value=0).to_numpy()
        )
    return schedule


def _exposure_schedule(exposure: pd.Series | None, dates: pd.DatetimeIndex) -> dict:
    schedule = {}
    if exposure is None:
        return schedule
    if (
        not isinstance(exposure.index, pd.DatetimeIndex)
        or exposure.index.has_duplicates
        or exposure.index.hasnans
        or not np.isfinite(exposure).all()
        or not exposure.gt(0).all()
        or not exposure.le(1).all()
    ):
        raise ValueError("exposure needs unique dates and finite values in (0, 1]")
    for decision, value in exposure.items():
        if not isinstance(decision, pd.Timestamp):
            raise ValueError("exposure decisions must be timestamps")
        idx = int(np.searchsorted(dates.to_numpy(), decision.to_datetime64(), side="right"))
        if idx < len(dates):
            if idx in schedule:
                raise ValueError("multiple exposure decisions map to one execution close")
            schedule[idx] = float(value)
    return schedule


def _post_cost_nav(pre_nav: float, weights: np.ndarray, holdings: np.ndarray, fee: float) -> float:
    """Solve the self-financing target value including both trade directions."""
    low, high = 0.0, pre_nav
    for _ in range(60):
        post_nav = (low + high) / 2
        residual = post_nav + fee * np.abs(weights * post_nav - holdings).sum() - pre_nav
        if residual > 0:
            high = post_nav
        else:
            low = post_nav
    return (low + high) / 2


def _log_holdings(log, date, symbols, shares, prices, nav):
    if log is not None:
        log.extend(
            {
                "trade_date": date,
                "symbol": symbol,
                "adjusted_units": float(units),
                "market_value": float(units * price),
                "weight": float(units * price / nav),
            }
            for symbol, units, price in zip(symbols, shares, prices, strict=True)
            if units > 0
        )


def _block_frame(frame, prices, name):
    if frame is None:
        return np.zeros(prices.shape, dtype=bool)
    if (
        not frame.index.equals(prices.index)
        or not frame.columns.equals(prices.columns)
        or frame.isna().any().any()
        or not all(pd.api.types.is_bool_dtype(dtype) for dtype in frame.dtypes)
    ):
        raise ValueError(f"{name} must be an aligned complete boolean frame")
    return frame.to_numpy(dtype=bool)


def _daily_record(date, nav, cash, previous_nav, pre_nav, cost, buys, sells, deferred):
    return {
        "trade_date": date,
        "nav": nav,
        "cash": cash,
        "net_return": nav / previous_nav - 1,
        "gross_return": pre_nav / previous_nav - 1,
        "cost": cost,
        "buy_notional": buys,
        "sell_notional": sells,
        "traded_fraction": (buys + sells) / pre_nav,
        "rebalance_deferred": deferred,
    }


def replay_close_targets(
    targets: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    cost_bps: float = 25.0,
    suspended: pd.DataFrame | None = None,
    exposure: pd.Series | None = None,
    holdings_log: list[dict] | None = None,
    buy_blocked: pd.DataFrame | None = None,
    sell_blocked: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Replay next-close targets with explicit all-or-defer suspension execution.

    A confirmed suspension carries the last observed mark for existing shares.
    The entire rebalance waits if any held/target name is suspended; newer
    targets supersede pending targets. This is a conservative execution policy,
    not a simulation of partial fills. Unknown price gaps still raise errors.
    Directional trade blocks also defer the whole basket, but never replace
    observed valuation marks. They apply only to nonzero proposed trades.

    Optional exposure orders are indexed by decision close and execute strictly
    later. Values must be in (0, 1]. Between stock selections an exposure change
    scales the drifted basket, not its original weights. On stock-selection
    dates it scales the new targets. Full liquidation/reentry requires explicit
    stock targets instead of this positive-exposure interface.
    """
    if not 0 <= cost_bps < 10000:
        raise ValueError("cost_bps must be in [0, 10000)")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("prices require a DatetimeIndex")
    schedule = _execution_schedule(targets, prices)
    exposure_schedule = _exposure_schedule(exposure, prices.index)
    buy_blocks = _block_frame(buy_blocked, prices, "buy_blocked")
    sell_blocks = _block_frame(sell_blocked, prices, "sell_blocked")
    suspension_flags = _block_frame(suspended, prices, "suspended")
    shares = np.zeros(len(prices.columns))
    cash = 1.0
    previous_nav = 1.0
    fee = cost_bps / 10000
    rows = []
    last_marks = np.full(len(prices.columns), np.nan)
    pending = None
    desired_exposure = 1.0
    exposure_pending = False
    for idx, (date, row) in enumerate(prices.iterrows()):
        px = row.to_numpy(dtype=float)
        halted = suspension_flags[idx]
        observed = np.isfinite(px) & (px > 0) & ~halted
        last_marks[observed] = px[observed]
        px = np.where(halted, last_marks, px)
        if idx in schedule:
            pending = schedule[idx]
        if idx in exposure_schedule:
            desired_exposure = exposure_schedule[idx]
            exposure_pending = True
        required = shares > 0
        if pending is not None:
            required = required | ((pending > 0) & ~halted)
        if not np.isfinite(px[required]).all() or (px[required] <= 0).any():
            raise ValueError(f"held/traded prices must be complete, finite and positive on {date}")
        # Placeholders only for zero holdings and zero targets, never for exposure.
        px = np.where(required, px, 1.0)
        holdings = shares * px
        pre_nav = float(holdings.sum() + cash)
        candidate = pending
        if exposure_pending and candidate is None and holdings.sum() > 0:
            candidate = holdings / holdings.sum()
        buys = sells = cost = 0.0
        deferred = candidate is not None and bool((halted & ((shares > 0) | (candidate > 0))).any())
        if candidate is not None and not deferred:
            weights = candidate * desired_exposure
            post_nav = _post_cost_nav(pre_nav, weights, holdings, fee)
            trades = weights * post_nav - holdings
            deferred = bool(
                ((trades > 1e-12) & buy_blocks[idx]).any()
                or ((trades < -1e-12) & sell_blocks[idx]).any()
            )
            if not deferred:
                buys = float(np.maximum(trades, 0).sum())
                sells = float(np.maximum(-trades, 0).sum())
                cost = fee * (buys + sells)
                shares = weights * post_nav / px
                cash = post_nav * (1 - weights.sum())
                pending = None
                exposure_pending = False
        nav = float((shares * px).sum() + cash)
        _log_holdings(holdings_log, date, prices.columns, shares, px, nav)
        rows.append(
            _daily_record(date, nav, cash, previous_nav, pre_nav, cost, buys, sells, deferred)
        )
        previous_nav = nav
    return pd.DataFrame(rows)
