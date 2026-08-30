"""Public execution diagnostics derived from canonical order and fill tables."""

from __future__ import annotations

import pandas as pd


def attribute_delayed_fills(
    orders: pd.DataFrame,
    fills: pd.DataFrame,
    pricing: pd.DataFrame,
) -> pd.DataFrame:
    """Attribute fill delay, price movement, and temporary impact per order.

    ``delay_opportunity_cost`` is positive when the first fill moved against the
    requested side and negative when delayed execution helped. It is an
    execution diagnostic, not alpha attribution.
    """

    required_orders = {
        "rebalance_date",
        "entry_date",
        "symbol",
        "side",
        "requested_notional",
        "filled_notional",
        "unfilled_notional",
    }
    required_pricing = {"trade_date", "symbol", "close"}
    missing_orders = sorted(required_orders - set(orders.columns))
    missing_pricing = sorted(required_pricing - set(pricing.columns))
    if missing_orders:
        raise ValueError("orders are missing required columns: " + ", ".join(missing_orders))
    if missing_pricing:
        raise ValueError("pricing is missing required columns: " + ", ".join(missing_pricing))

    keys = ["rebalance_date", "entry_date", "symbol", "side"]
    out = orders[[*keys, "requested_notional", "filled_notional", "unfilled_notional"]].copy()
    for column in ("rebalance_date", "entry_date"):
        out[column] = pd.to_datetime(out[column], errors="coerce").dt.normalize()
    out["symbol"] = out["symbol"].astype(str)
    out["side"] = out["side"].astype(str).str.lower()

    fill_work = fills.copy()
    if fill_work.empty:
        first = pd.DataFrame(columns=[*keys, "first_fill_date", "temporary_impact"])
    else:
        fill_work["trade_date"] = pd.to_datetime(
            fill_work["trade_date"], errors="coerce"
        ).dt.normalize()
        for column in ("rebalance_date", "entry_date"):
            fill_work[column] = pd.to_datetime(fill_work[column], errors="coerce").dt.normalize()
        fill_work["symbol"] = fill_work["symbol"].astype(str)
        fill_work["side"] = fill_work["side"].astype(str).str.lower()
        first = (
            fill_work.sort_values("trade_date")
            .groupby(keys, as_index=False, sort=False)
            .agg(
                first_fill_date=("trade_date", "first"),
                temporary_impact=("cost_temporary_impact", "sum"),
            )
        )

    out = out.merge(first, on=keys, how="left")
    out["delay_days"] = (out["first_fill_date"] - out["entry_date"]).dt.days.fillna(0).astype(int)

    price_work = pricing[["trade_date", "symbol", "close"]].copy()
    price_work["trade_date"] = pd.to_datetime(
        price_work["trade_date"], errors="coerce"
    ).dt.normalize()
    price_work["close"] = pd.to_numeric(price_work["close"], errors="coerce")
    entry_price = price_work.rename(columns={"trade_date": "entry_date", "close": "entry_price"})
    fill_price = price_work.rename(
        columns={"trade_date": "first_fill_date", "close": "first_fill_price"}
    )
    out = out.merge(entry_price, on=["entry_date", "symbol"], how="left")
    out = out.merge(fill_price, on=["first_fill_date", "symbol"], how="left")

    raw_return = out["first_fill_price"].div(out["entry_price"]) - 1.0
    side_sign = out["side"].map({"buy": 1.0, "sell": -1.0}).fillna(0.0)
    out["reference_return_to_first_fill"] = (raw_return * side_sign).fillna(0.0)
    out["delay_opportunity_cost"] = (
        out["unfilled_notional"] * out["reference_return_to_first_fill"]
    ).astype(float)
    out["temporary_impact"] = out["temporary_impact"].fillna(0.0)
    return out


__all__ = ["attribute_delayed_fills"]
