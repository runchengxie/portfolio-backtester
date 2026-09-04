from pathlib import Path
from typing import cast

import pandas as pd
import pytest

from portfolio_backtester.allocation_selection import (
    prepare_selection,
    select_latest_holdings,
    select_from_positions_file,
)


def test_select_from_positions_file_uses_latest_eligible_entry(tmp_path: Path) -> None:
    path = tmp_path / "positions.csv"
    pd.DataFrame(
        {
            "symbol": ["OLD", "NEW"],
            "entry_date": ["2026-01-02", "2026-01-03"],
            "weight": [0.5, 1.0],
        }
    ).to_csv(path, index=False)

    selection, entry_date = select_from_positions_file(
        path,
        cast(pd.Timestamp, pd.Timestamp("2026-01-03")),
    )

    assert entry_date == pd.Timestamp("2026-01-03")
    assert selection["symbol"].tolist() == ["NEW"]


def test_select_from_positions_file_can_select_latest_future_entry(tmp_path: Path) -> None:
    path = tmp_path / "positions.csv"
    pd.DataFrame(
        {
            "symbol": ["OLD", "NEW"],
            "entry_date": ["2026-01-02", "2026-01-03"],
            "weight": [0.5, 1.0],
        }
    ).to_csv(path, index=False)

    selection, entry_date = select_from_positions_file(
        path,
        cast(pd.Timestamp, pd.Timestamp("2026-01-01")),
        allow_future_entry=True,
    )

    assert entry_date == pd.Timestamp("2026-01-03")
    assert selection["symbol"].tolist() == ["NEW"]


def test_prepare_selection_filters_side_and_rank() -> None:
    selection = pd.DataFrame(
        {
            "symbol": ["B", "A", "C"],
            "side": ["long", "long", "short"],
            "rank": [2, 1, 1],
            "order_book_id": ["B", "A", "C"],
        }
    )

    prepared = prepare_selection(selection, side="long", top_n=1)

    assert prepared["symbol"].tolist() == ["A"]
    assert "order_book_id" not in prepared


def test_select_latest_holdings_accepts_an_in_memory_frame() -> None:
    selection, entry_date = select_latest_holdings(
        pd.DataFrame(
            {
                "symbol": ["OLD", "NEW"],
                "entry_date": ["2026-01-02", "2026-01-03"],
            }
        ),
        cast(pd.Timestamp, pd.Timestamp("2026-01-03")),
    )

    assert entry_date == pd.Timestamp("2026-01-03")
    assert selection["symbol"].tolist() == ["NEW"]


def test_prepare_selection_rejects_empty_side() -> None:
    with pytest.raises(SystemExit, match="side=short"):
        prepare_selection(pd.DataFrame({"symbol": ["A"]}), side="short", top_n=1)
