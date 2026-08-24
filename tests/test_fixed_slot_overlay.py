from __future__ import annotations

import pandas as pd
import pytest

from portfolio_backtester.fixed_slot_overlay import (
    FixedSlotOverlayError,
    build_fixed_slot_overlay_target,
)


def _ranking() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": "A",
                "is_current_holding": True,
                "hold_eligible": True,
                "entry_eligible": False,
                "in_entry_pool": False,
                "overlay_rank": 2,
            },
            {
                "symbol": "B",
                "is_current_holding": True,
                "hold_eligible": False,
                "entry_eligible": False,
                "in_entry_pool": False,
                "overlay_rank": 8,
            },
            {
                "symbol": "C",
                "is_current_holding": False,
                "hold_eligible": False,
                "entry_eligible": True,
                "in_entry_pool": True,
                "overlay_rank": 1,
            },
            {
                "symbol": "D",
                "is_current_holding": False,
                "hold_eligible": False,
                "entry_eligible": True,
                "in_entry_pool": True,
                "overlay_rank": 3,
            },
        ]
    )


def test_build_fixed_slot_overlay_target_retains_adds_and_accounts_turnover() -> None:
    target = build_fixed_slot_overlay_target(
        _ranking(),
        ("A", "B"),
        target_slots=3,
        retain_rank_lte=5,
        new_entry_rank_lte=3,
    )

    assert target.target_symbols == ("A", "C", "D")
    assert target.retained_symbols == ("A",)
    assert target.new_symbols == ("C", "D")
    assert target.exited_symbols == ("B",)
    assert target.target_weights.to_dict() == pytest.approx(
        {"A": 1.0 / 3.0, "C": 1.0 / 3.0, "D": 1.0 / 3.0}
    )
    assert target.target_cash_weight == pytest.approx(0.0)
    assert target.target_name_turnover == pytest.approx(0.5)
    assert target.target_full_l1 == pytest.approx(1.0)
    assert target.target_half_l1 == pytest.approx(0.5)


def test_build_fixed_slot_overlay_target_rejects_duplicate_incumbents() -> None:
    with pytest.raises(FixedSlotOverlayError, match="duplicates"):
        build_fixed_slot_overlay_target(
            _ranking(),
            ("A", "A"),
            target_slots=3,
            retain_rank_lte=5,
            new_entry_rank_lte=3,
        )


def test_build_fixed_slot_overlay_target_rejects_unknown_flagged_incumbent() -> None:
    with pytest.raises(FixedSlotOverlayError, match="absent from incumbent_symbols"):
        build_fixed_slot_overlay_target(
            _ranking(),
            ("B",),
            target_slots=3,
            retain_rank_lte=5,
            new_entry_rank_lte=3,
        )
