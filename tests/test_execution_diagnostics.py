from __future__ import annotations

import pandas as pd
import pytest

from portfolio_backtester import attribute_delayed_fills


def test_attribute_delayed_fills_separates_delay_and_impact_for_buy() -> None:
    orders = pd.DataFrame(
        [
            {
                "rebalance_date": "20240102",
                "entry_date": "20240103",
                "symbol": "AAA",
                "side": "buy",
                "requested_notional": 1000.0,
                "filled_notional": 600.0,
                "unfilled_notional": 400.0,
            }
        ]
    )
    fills = pd.DataFrame(
        [
            {
                "rebalance_date": "20240102",
                "entry_date": "20240103",
                "symbol": "AAA",
                "side": "buy",
                "trade_date": "20240104",
                "filled_notional": 600.0,
                "cost_temporary_impact": 6.0,
            }
        ]
    )
    pricing = pd.DataFrame(
        [
            {"trade_date": "20240103", "symbol": "AAA", "close": 10.0},
            {"trade_date": "20240104", "symbol": "AAA", "close": 9.0},
        ]
    )

    attribution = attribute_delayed_fills(orders, fills, pricing)

    row = attribution.iloc[0]
    assert row["delay_days"] == 1
    assert row["reference_return_to_first_fill"] == pytest.approx(-0.1)
    assert row["delay_opportunity_cost"] == pytest.approx(-40.0)
    assert row["temporary_impact"] == pytest.approx(6.0)


def test_attribute_delayed_fills_flips_price_move_for_sell() -> None:
    orders = pd.DataFrame(
        [
            {
                "rebalance_date": "20240102",
                "entry_date": "20240103",
                "symbol": "AAA",
                "side": "sell",
                "requested_notional": 1000.0,
                "filled_notional": 600.0,
                "unfilled_notional": 400.0,
            }
        ]
    )
    fills = pd.DataFrame(
        [
            {
                "rebalance_date": "20240102",
                "entry_date": "20240103",
                "symbol": "AAA",
                "side": "sell",
                "trade_date": "20240104",
                "filled_notional": 600.0,
                "cost_temporary_impact": 4.0,
            }
        ]
    )
    pricing = pd.DataFrame(
        [
            {"trade_date": "20240103", "symbol": "AAA", "close": 10.0},
            {"trade_date": "20240104", "symbol": "AAA", "close": 11.0},
        ]
    )

    attribution = attribute_delayed_fills(orders, fills, pricing)

    row = attribution.iloc[0]
    assert row["reference_return_to_first_fill"] == pytest.approx(-0.1)
    assert row["delay_opportunity_cost"] == pytest.approx(-40.0)
    assert row["temporary_impact"] == pytest.approx(4.0)


def test_attribute_delayed_fills_handles_orders_without_fills() -> None:
    orders = pd.DataFrame(
        [
            {
                "rebalance_date": "20240102",
                "entry_date": "20240103",
                "symbol": "AAA",
                "side": "buy",
                "requested_notional": 1000.0,
                "filled_notional": 0.0,
                "unfilled_notional": 1000.0,
            }
        ]
    )
    pricing = pd.DataFrame([{"trade_date": "20240103", "symbol": "AAA", "close": 10.0}])

    attribution = attribute_delayed_fills(orders, pd.DataFrame(), pricing)

    row = attribution.iloc[0]
    assert pd.isna(row["first_fill_date"])
    assert row["delay_days"] == 0
    assert row["reference_return_to_first_fill"] == 0.0
    assert row["delay_opportunity_cost"] == 0.0
    assert row["temporary_impact"] == 0.0


def test_attribute_delayed_fills_requires_pricing_columns() -> None:
    orders = pd.DataFrame(
        [
            {
                "rebalance_date": "20240102",
                "entry_date": "20240103",
                "symbol": "AAA",
                "side": "buy",
                "requested_notional": 1000.0,
                "filled_notional": 0.0,
                "unfilled_notional": 1000.0,
            }
        ]
    )

    with pytest.raises(ValueError, match="pricing is missing required columns"):
        attribute_delayed_fills(orders, pd.DataFrame(), pd.DataFrame())
