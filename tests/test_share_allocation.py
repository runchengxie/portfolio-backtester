import pandas as pd
import pytest

from portfolio_backtester.share_allocation import allocate_equal_weight_shares


def test_allocate_equal_weight_shares_respects_round_lots_and_cash_buffer():
    selection = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "order_book_id": ["AAA", "BBB"],
            "side": ["long", "long"],
            "rank": [1, 2],
            "price": [10.0, 25.0],
            "round_lot": [100, 10],
        }
    )

    result = allocate_equal_weight_shares(
        selection,
        cash=10_000.0,
        buffer_bps=100.0,
    )

    assert result.investable_cash == 9_900.0
    assert result.allocations["target_value"].tolist() == [4_950.0, 4_950.0]
    assert result.allocations["lots"].tolist() == [4, 19]
    assert result.allocations["shares"].tolist() == [400, 190]
    assert result.allocations["est_value"].tolist() == [4_000.0, 4_750.0]
    assert result.est_total == 8_750.0
    assert result.cash_left == 1_150.0


def test_allocate_equal_weight_shares_preserves_metadata_columns():
    selection = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "order_book_id": ["AAA.X"],
            "side": ["short"],
            "rank": [3],
            "price": [20.0],
            "round_lot": [5],
        }
    )

    result = allocate_equal_weight_shares(selection, cash=1_000.0)

    row = result.allocations.iloc[0]
    assert row["symbol"] == "AAA"
    assert row["order_book_id"] == "AAA.X"
    assert row["side"] == "short"
    assert row["rank"] == 3
    assert row["round_lot"] == 5


def test_allocate_equal_weight_shares_rejects_empty_selection():
    with pytest.raises(ValueError, match="No holdings selected"):
        allocate_equal_weight_shares(
            pd.DataFrame(columns=["symbol", "price", "round_lot"]),
            cash=1_000.0,
        )


def test_allocate_equal_weight_shares_rejects_missing_reference_columns():
    selection = pd.DataFrame({"symbol": ["AAA"], "price": [10.0]})

    with pytest.raises(ValueError, match="round_lot"):
        allocate_equal_weight_shares(selection, cash=1_000.0)


def test_allocate_equal_weight_shares_rejects_nonpositive_price_or_lot():
    bad_price = pd.DataFrame({"symbol": ["AAA"], "price": [0.0], "round_lot": [100]})
    bad_lot = pd.DataFrame({"symbol": ["AAA"], "price": [10.0], "round_lot": [0]})

    with pytest.raises(ValueError, match="price"):
        allocate_equal_weight_shares(bad_price, cash=1_000.0)
    with pytest.raises(ValueError, match="round_lot"):
        allocate_equal_weight_shares(bad_lot, cash=1_000.0)
