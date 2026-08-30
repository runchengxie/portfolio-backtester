from __future__ import annotations

import pandas as pd
import pytest

from portfolio_backtester import settle_execution_fills


def test_settle_execution_fills_applies_fees_lots_and_t1() -> None:
    fills = pd.DataFrame(
        [
            {
                "trade_date": "2024-01-02",
                "instrument_id": "000001.SZ",
                "side": "buy",
                "filled_notional": 1_005.0,
                "average_fill_price": 10.05,
            },
            {
                "trade_date": "2024-01-02",
                "instrument_id": "000002.SZ",
                "side": "sell",
                "filled_notional": 500.0,
                "average_fill_price": 10.0,
            },
            {
                "trade_date": "2024-01-03",
                "instrument_id": "000001.SZ",
                "side": "sell",
                "filled_notional": 500.0,
                "average_fill_price": 10.0,
            },
        ]
    )
    marks = pd.DataFrame(
        [
            {"trade_date": "2024-01-02", "instrument_id": "000001.SZ", "price": 10.0},
            {"trade_date": "2024-01-03", "instrument_id": "000001.SZ", "price": 10.0},
        ]
    )

    result = settle_execution_fills(
        fills,
        marks,
        initial_capital=2_000.0,
        round_lot=100,
        buy_fee_bps=1.0,
        sell_fee_bps=1.0,
        stamp_tax_bps=5.0,
    )

    day_one = result.loc[result["trade_date"].eq(pd.Timestamp("2024-01-02"))].iloc[0]
    day_two = result.loc[result["trade_date"].eq(pd.Timestamp("2024-01-03"))].iloc[0]
    assert day_one["buy_shares"] == 100
    assert day_one["sell_shares"] == 0
    assert day_one["cash_end"] < 2_000.0
    assert day_two["sell_shares"] == 50
    assert day_two["t1_blocked_shares"] == 0


def test_settle_execution_fills_blocks_same_day_buys_from_being_sold() -> None:
    fills = pd.DataFrame(
        [
            {
                "trade_date": "2024-01-02",
                "instrument_id": "000001.SZ",
                "side": "buy",
                "filled_notional": 1_000.0,
                "average_fill_price": 10.0,
            },
            {
                "trade_date": "2024-01-02",
                "instrument_id": "000001.SZ",
                "side": "sell",
                "filled_notional": 1_000.0,
                "average_fill_price": 10.0,
            },
        ]
    )
    marks = pd.DataFrame(
        [{"trade_date": "2024-01-02", "instrument_id": "000001.SZ", "price": 10.0}]
    )

    result = settle_execution_fills(fills, marks, initial_capital=2_000.0, round_lot=100)

    assert result.loc[0, "sell_shares"] == 0
    assert result.loc[0, "t1_blocked_shares"] == 100


def test_settle_execution_fills_blocks_buys_when_cash_is_insufficient() -> None:
    fills = pd.DataFrame(
        [
            {
                "trade_date": "2024-01-02",
                "instrument_id": "000001.SZ",
                "side": "buy",
                "filled_notional": 2_000.0,
                "average_fill_price": 10.0,
            }
        ]
    )
    marks = pd.DataFrame(
        [{"trade_date": "2024-01-02", "instrument_id": "000001.SZ", "price": 10.0}]
    )

    result = settle_execution_fills(
        fills,
        marks,
        initial_capital=500.0,
        round_lot=100,
        buy_fee_bps=1.0,
    )

    assert result.loc[0, "buy_shares"] == 0
    assert result.loc[0, "cash_end"] == 500.0
    assert result.loc[0, "cash_blocked_shares"] == 200
    assert result.loc[0, "cash_blocked_notional"] == 2_000.0


def test_settle_execution_fills_validates_inputs() -> None:
    with pytest.raises(ValueError, match="initial_capital"):
        settle_execution_fills(pd.DataFrame(), pd.DataFrame(), initial_capital=0)
