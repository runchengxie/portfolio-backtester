from pathlib import Path

import pandas as pd

from portfolio_backtester.allocation_service import build_equal_weight_allocation


def test_build_equal_weight_allocation_composes_owner_helpers(tmp_path: Path) -> None:
    positions = pd.DataFrame(
        {
            "symbol": ["0001.HK", "0002.HK"],
            "entry_date": ["2026-01-02", "2026-01-02"],
            "side": ["long", "long"],
            "rank": [1, 2],
        }
    )
    reference = tmp_path / "reference.csv"
    pd.DataFrame(
        {
            "symbol": ["0001.HK", "0002.HK"],
            "price": [50.0, 20.0],
            "round_lot": [100, 200],
            "price_date": ["2026-01-02", "2026-01-02"],
        }
    ).to_csv(reference, index=False)

    result = build_equal_weight_allocation(
        positions,
        reference_file=reference,
        top_n=2,
        cash=100_000,
    )

    assert result.reference.price_date == "2026-01-02"
    assert result.allocations["symbol"].tolist() == ["0001.HK", "0002.HK"]
    assert result.estimated_value > 0
