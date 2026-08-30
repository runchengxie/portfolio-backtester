"""Cash-ledger settlement for broker-independent historical execution fills."""

from __future__ import annotations

import numpy as np
import pandas as pd


def settle_execution_fills(
    fills: pd.DataFrame,
    marks: pd.DataFrame,
    *,
    initial_capital: float,
    round_lot: int = 100,
    buy_fee_bps: float = 0.0,
    sell_fee_bps: float = 0.0,
    stamp_tax_bps: float = 0.0,
) -> pd.DataFrame:
    """Settle fills into daily cash, holdings, T+1 constraints, and marked NAV.

    Fills must contain ``trade_date``, ``instrument_id``, ``side``,
    ``filled_notional`` and ``average_fill_price``. Marks contain
    ``trade_date``, ``instrument_id`` and ``price``. Buys are rounded down to
    ``round_lot`` shares; sells may use odd lots but cannot exceed inventory
    available before the current trading date.
    """

    if not np.isfinite(initial_capital) or initial_capital <= 0:
        raise ValueError("initial_capital must be positive")
    if round_lot <= 0:
        raise ValueError("round_lot must be positive")
    for value, name in (
        (buy_fee_bps, "buy_fee_bps"),
        (sell_fee_bps, "sell_fee_bps"),
        (stamp_tax_bps, "stamp_tax_bps"),
    ):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be non-negative")

    required_fills = {
        "trade_date",
        "instrument_id",
        "side",
        "filled_notional",
        "average_fill_price",
    }
    required_marks = {"trade_date", "instrument_id", "price"}
    missing_fills = sorted(required_fills - set(fills.columns))
    missing_marks = sorted(required_marks - set(marks.columns))
    if missing_fills:
        raise ValueError(f"fills missing columns: {missing_fills}")
    if missing_marks:
        raise ValueError(f"marks missing columns: {missing_marks}")

    fill_frame = fills.copy()
    fill_frame["trade_date"] = pd.to_datetime(fill_frame["trade_date"]).dt.normalize()
    fill_frame["instrument_id"] = fill_frame["instrument_id"].astype(str)
    fill_frame["side"] = fill_frame["side"].astype(str).str.lower()
    if not fill_frame["side"].isin(["buy", "sell"]).all():
        raise ValueError("fills side must be buy or sell")
    fill_frame["filled_notional"] = pd.to_numeric(
        fill_frame["filled_notional"], errors="raise"
    )
    fill_frame["average_fill_price"] = pd.to_numeric(
        fill_frame["average_fill_price"], errors="raise"
    )

    mark_frame = marks.copy()
    mark_frame["trade_date"] = pd.to_datetime(mark_frame["trade_date"]).dt.normalize()
    mark_frame["instrument_id"] = mark_frame["instrument_id"].astype(str)
    mark_frame["price"] = pd.to_numeric(mark_frame["price"], errors="raise")

    holdings: dict[str, int] = {}
    cash = float(initial_capital)
    rows: list[dict[str, float | int | pd.Timestamp]] = []

    for trade_date in sorted(set(fill_frame["trade_date"]) | set(mark_frame["trade_date"])):
        day_fills = fill_frame.loc[fill_frame["trade_date"].eq(trade_date)]
        # Sell proceeds can fund same-day buys. Same-day buys remain unavailable
        # for sale under A-share T+1, so process sells first deterministically.
        day_fills = pd.concat(
            [
                day_fills.loc[day_fills["side"].eq("sell")],
                day_fills.loc[day_fills["side"].eq("buy")],
            ],
            ignore_index=True,
        )
        available_at_open = holdings.copy()
        buy_notional = sell_notional = fees = 0.0
        buy_shares = sell_shares = blocked_shares = 0
        lot_blocked_shares = cash_blocked_shares = 0
        lot_blocked_notional = cash_blocked_notional = 0.0

        for fill in day_fills.itertuples(index=False):
            requested_shares = int(
                np.floor(float(fill.filled_notional) / float(fill.average_fill_price))
            )
            if fill.side == "buy":
                lot_shares = (requested_shares // round_lot) * round_lot
                affordable_shares = int(
                    np.floor(
                        cash
                        / (
                            float(fill.average_fill_price)
                            * (1.0 + buy_fee_bps / 10000.0)
                        )
                    )
                )
                affordable_shares = (affordable_shares // round_lot) * round_lot
                shares = min(lot_shares, affordable_shares)
                lot_blocked_shares += max(requested_shares - lot_shares, 0)
                cash_blocked_shares += max(lot_shares - shares, 0)
                notional = shares * float(fill.average_fill_price)
                fee = notional * buy_fee_bps / 10000.0
                lot_blocked_notional += max(
                    float(fill.filled_notional)
                    - lot_shares * float(fill.average_fill_price),
                    0.0,
                )
                cash_blocked_notional += max(
                    (lot_shares - shares) * float(fill.average_fill_price),
                    0.0,
                )
                cash -= notional + fee
                holdings[fill.instrument_id] = holdings.get(fill.instrument_id, 0) + shares
                buy_shares += shares
                buy_notional += notional
            else:
                available = min(
                    holdings.get(fill.instrument_id, 0),
                    available_at_open.get(fill.instrument_id, 0),
                )
                shares = min(requested_shares, available)
                blocked_shares += max(requested_shares - shares, 0)
                notional = shares * float(fill.average_fill_price)
                fee = notional * (sell_fee_bps + stamp_tax_bps) / 10000.0
                cash += notional - fee
                holdings[fill.instrument_id] = holdings.get(fill.instrument_id, 0) - shares
                sell_shares += shares
                sell_notional += notional
            fees += fee

        mark_day = mark_frame.loc[mark_frame["trade_date"].eq(trade_date)]
        mark_prices = dict(zip(mark_day["instrument_id"], mark_day["price"], strict=True))
        holdings_value = sum(
            shares * float(mark_prices.get(symbol, 0.0)) for symbol, shares in holdings.items()
        )
        nav = cash + holdings_value
        rows.append(
            {
                "trade_date": trade_date,
                "cash_end": cash,
                "holdings_value": holdings_value,
                "nav": nav,
                "cash_weight": cash / nav if nav else np.nan,
                "buy_notional": buy_notional,
                "sell_notional": sell_notional,
                "buy_shares": buy_shares,
                "sell_shares": sell_shares,
                "t1_blocked_shares": blocked_shares,
                "lot_blocked_shares": lot_blocked_shares,
                "lot_blocked_notional": lot_blocked_notional,
                "cash_blocked_shares": cash_blocked_shares,
                "cash_blocked_notional": cash_blocked_notional,
                "fees": fees,
            }
        )
    return pd.DataFrame(rows)


__all__ = ["settle_execution_fills"]
