from __future__ import annotations

import pandas as pd
import pytest

from portfolio_backtester.allocation_reference import (
    join_allocation_reference,
    load_allocation_reference,
)


def _write_reference(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def test_load_allocation_reference_normalizes_required_columns(tmp_path):
    path = tmp_path / "reference.csv"
    _write_reference(
        path,
        [
            {
                "symbol": " 00001.HK ",
                "price": 50.0,
                "round_lot": 100,
                "price_date": "2020-01-03",
                "order_book_id": "00001.XHKG",
                "source": "fixture",
            }
        ],
    )

    reference = load_allocation_reference(path)

    row = reference.frame.iloc[0]
    assert row["symbol"] == "00001.HK"
    assert row["price"] == 50.0
    assert row["round_lot"] == 100
    assert row["price_date"] == "2020-01-03"
    assert row["order_book_id"] == "00001.XHKG"
    assert reference.price_date == "2020-01-03"
    assert reference.source == "fixture"


def test_load_allocation_reference_supports_parquet(tmp_path):
    path = tmp_path / "reference.parquet"
    pd.DataFrame(
        {
            "symbol": ["600519.SH"],
            "price": [100.0],
            "round_lot": [100],
            "price_date": ["2026-09-01"],
        }
    ).to_parquet(path, index=False)

    reference = load_allocation_reference(path)

    assert reference.frame["symbol"].tolist() == ["600519.SH"]
    assert reference.source == "reference_file"


def test_load_allocation_reference_rejects_missing_or_bad_fields(tmp_path):
    missing = tmp_path / "missing.csv"
    pd.DataFrame({"symbol": ["AAA"], "price": [10.0]}).to_csv(missing, index=False)

    with pytest.raises(SystemExit, match="round_lot"):
        load_allocation_reference(missing)

    bad = tmp_path / "bad.csv"
    pd.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "price": [0.0, 10.0],
            "round_lot": [100, -1],
            "price_date": ["2026-09-01", "2026-09-01"],
        }
    ).to_csv(bad, index=False)

    with pytest.raises(SystemExit, match="positive"):
        load_allocation_reference(bad)


def test_load_allocation_reference_rejects_duplicate_symbols_and_mixed_dates(tmp_path):
    duplicate = tmp_path / "duplicate.csv"
    pd.DataFrame(
        {
            "symbol": ["AAA", "AAA"],
            "price": [10.0, 11.0],
            "round_lot": [100, 100],
            "price_date": ["2026-09-01", "2026-09-01"],
        }
    ).to_csv(duplicate, index=False)

    with pytest.raises(SystemExit, match="duplicate symbol"):
        load_allocation_reference(duplicate)

    mixed = tmp_path / "mixed.csv"
    pd.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "price": [10.0, 11.0],
            "round_lot": [100, 100],
            "price_date": ["2026-09-01", "2026-09-02"],
        }
    ).to_csv(mixed, index=False)

    with pytest.raises(SystemExit, match="single price_date"):
        load_allocation_reference(mixed)


def test_join_allocation_reference_preserves_selection_order(tmp_path):
    path = tmp_path / "reference.csv"
    _write_reference(
        path,
        [
            {"symbol": "BBB", "price": 20.0, "round_lot": 200, "price_date": "2026-09-01"},
            {"symbol": "AAA", "price": 10.0, "round_lot": 100, "price_date": "2026-09-01"},
        ],
    )
    reference = load_allocation_reference(path)
    selection = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "side": ["long", "long"],
            "rank": [1, 2],
        }
    )

    joined = join_allocation_reference(selection, reference)

    assert joined["symbol"].tolist() == ["AAA", "BBB"]
    assert joined["price"].tolist() == [10.0, 20.0]
    assert joined["round_lot"].tolist() == [100, 200]


def test_join_allocation_reference_rejects_missing_selected_symbol(tmp_path):
    path = tmp_path / "reference.csv"
    _write_reference(
        path,
        [{"symbol": "AAA", "price": 10.0, "round_lot": 100, "price_date": "2026-09-01"}],
    )
    reference = load_allocation_reference(path)
    selection = pd.DataFrame({"symbol": ["AAA", "BBB"]})

    with pytest.raises(SystemExit, match="BBB"):
        join_allocation_reference(selection, reference)
