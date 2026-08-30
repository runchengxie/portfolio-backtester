from __future__ import annotations

import pandas as pd
import pytest

from portfolio_backtester import positions_by_rebalance_from_targets


def test_positions_by_rebalance_from_targets_normalizes_and_sorts_mapping() -> None:
    targets = {
        pd.Timestamp("2024-02-01"): {"BBB": 0.25, "AAA": 0.5},
        pd.Timestamp("2024-01-02"): {"AAA": 0.75},
    }

    positions = positions_by_rebalance_from_targets(
        targets,
        entry_dates={
            pd.Timestamp("2024-02-01"): pd.Timestamp("2024-02-02"),
            pd.Timestamp("2024-01-02"): pd.Timestamp("2024-01-03"),
        },
    )

    assert positions.to_dict("records") == [
        {
            "rebalance_date": pd.Timestamp("2024-01-02"),
            "entry_date": pd.Timestamp("2024-01-03"),
            "symbol": "AAA",
            "weight": 0.75,
            "side": "long",
        },
        {
            "rebalance_date": pd.Timestamp("2024-02-01"),
            "entry_date": pd.Timestamp("2024-02-02"),
            "symbol": "AAA",
            "weight": 0.5,
            "side": "long",
        },
        {
            "rebalance_date": pd.Timestamp("2024-02-01"),
            "entry_date": pd.Timestamp("2024-02-02"),
            "symbol": "BBB",
            "weight": 0.25,
            "side": "long",
        },
    ]


def test_positions_by_rebalance_from_targets_accepts_target_weight_alias() -> None:
    targets = pd.DataFrame(
        [
            {
                "rebalance_date": "20240102",
                "entry_date": "20240103",
                "symbol": " AAA ",
                "target_weight": 0.5,
            }
        ]
    )

    positions = positions_by_rebalance_from_targets(targets)

    assert positions.to_dict("records") == [
        {
            "rebalance_date": pd.Timestamp("2024-01-02"),
            "entry_date": pd.Timestamp("2024-01-03"),
            "symbol": "AAA",
            "weight": 0.5,
            "side": "long",
        }
    ]


def test_positions_by_rebalance_from_targets_rejects_duplicate_symbols() -> None:
    targets = pd.DataFrame(
        [
            {"rebalance_date": "20240102", "symbol": "AAA", "weight": 0.5},
            {"rebalance_date": "20240102", "symbol": "AAA", "weight": 0.5},
        ]
    )

    with pytest.raises(ValueError, match="duplicate"):
        positions_by_rebalance_from_targets(targets)


def test_positions_by_rebalance_from_targets_rejects_negative_weight() -> None:
    targets = pd.DataFrame([{"rebalance_date": "20240102", "symbol": "AAA", "weight": -0.1}])

    with pytest.raises(ValueError, match="non-negative"):
        positions_by_rebalance_from_targets(targets)


def test_positions_by_rebalance_from_targets_rejects_non_long_side() -> None:
    targets = pd.DataFrame(
        [
            {
                "rebalance_date": "20240102",
                "symbol": "AAA",
                "weight": 0.5,
                "side": "short",
            }
        ]
    )

    with pytest.raises(ValueError, match="only long targets"):
        positions_by_rebalance_from_targets(targets)
