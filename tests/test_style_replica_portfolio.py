from __future__ import annotations

import pandas as pd
import pytest

from cstree.backtesting.style_replica_portfolio import (
    StyleReplicaPortfolioConfig,
    build_style_replica_positions,
    compute_daily_changes,
    compute_style_exposure_summary,
)


def test_build_style_replica_positions_aggregates_overlap() -> None:
    signals = pd.DataFrame(
        [
            {
                "trade_date": "2026-01-02",
                "symbol": "AAA",
                "score_a": 10.0,
                "score_b": 9.0,
                "theme": "growth",
                "industry": "software",
            },
            {
                "trade_date": "2026-01-02",
                "symbol": "BBB",
                "score_a": 8.0,
                "score_b": 7.0,
                "theme": "growth",
                "industry": "hardware",
            },
        ]
    )
    config = StyleReplicaPortfolioConfig(
        a_slots=1,
        b_slots=1,
        theme_quotas={"growth": 1},
        normal_slot_weight=0.10,
        max_name_weight=0.20,
    )

    positions = build_style_replica_positions(signals, config=config)

    assert positions.shape[0] == 1
    assert positions.loc[0, "symbol"] == "AAA"
    assert positions.loc[0, "leg"] == "A+B"
    assert positions.loc[0, "weight"] == pytest.approx(0.20)
    assert positions.loc[0, "rebalance_date"] == "20260102"
    assert positions.loc[0, "entry_date"] == "20260102"
    assert positions.loc[0, "rank"] == 1


def test_compute_daily_changes_classifies_position_updates() -> None:
    positions = pd.DataFrame(
        [
            {"rebalance_date": "20260102", "symbol": "AAA", "weight": 0.5, "leg": "A"},
            {"rebalance_date": "20260102", "symbol": "BBB", "weight": 0.5, "leg": "B"},
            {"rebalance_date": "20260105", "symbol": "AAA", "weight": 0.4, "leg": "A"},
            {"rebalance_date": "20260105", "symbol": "CCC", "weight": 0.6, "leg": "B"},
        ]
    )

    changes = compute_daily_changes(positions)
    latest = changes.loc[changes["rebalance_date"] == "20260105"].set_index("symbol")

    assert latest.loc["AAA", "action"] == "weight_change"
    assert latest.loc["BBB", "action"] == "exit"
    assert latest.loc["CCC", "action"] == "new"
    assert latest.loc["AAA", "weight_change"] == pytest.approx(-0.1)


def test_compute_style_exposure_summary_counts_overlap() -> None:
    positions = pd.DataFrame(
        [
            {
                "rebalance_date": "20260102",
                "symbol": "AAA",
                "weight": 0.2,
                "leg": "A+B",
                "theme": "growth",
                "industry": "software",
            },
            {
                "rebalance_date": "20260102",
                "symbol": "BBB",
                "weight": 0.1,
                "leg": "A",
                "theme": "quality",
                "industry": "hardware",
            },
        ]
    )

    summary = compute_style_exposure_summary(positions)

    assert summary["total_stocks"] == 2
    assert summary["a_leg_count"] == 2
    assert summary["b_leg_count"] == 1
    assert summary["overlap_count"] == 1
    assert summary["total_weight"] == pytest.approx(0.3)
