from __future__ import annotations

import pandas as pd

from portfolio_backtester.sleeve_portfolio import (
    QuotaSleeveSpec,
    RankBufferedSleeveSpec,
    SleevePortfolioSpec,
    build_sleeve_positions,
    compute_position_changes,
    compute_position_exposure,
)


def _signals() -> pd.DataFrame:
    rows = [
        ("20260105", "000001", 9.0, None, "t1", "bank"),
        ("20260105", "000002", 8.0, 9.0, "t1", "bank"),
        ("20260105", "000003", 7.0, 8.0, "t1", "bank"),
        ("20260105", "000004", 9.0, 7.0, "t2", "tech"),
        ("20260105", "000005", 8.0, 6.0, "t2", "tech"),
        ("20260105", "000006", 7.0, 5.0, "t2", "health"),
        ("20260106", "000001", 9.0, None, "t1", "bank"),
        ("20260106", "000007", 8.5, 9.5, "t1", "tech"),
        ("20260106", "000002", 8.0, 9.0, "t1", "bank"),
        ("20260106", "000004", 9.0, 8.5, "t2", "tech"),
        ("20260106", "000005", 8.0, 8.0, "t2", "tech"),
        ("20260106", "000006", 7.0, 7.5, "t2", "health"),
    ]
    return pd.DataFrame(
        rows,
        columns=["signal_date", "symbol", "score_a", "score_b", "theme", "industry"],
    )


def _spec() -> SleevePortfolioSpec:
    return SleevePortfolioSpec(
        quota_sleeve=QuotaSleeveSpec(
            name="A",
            score_col="score_a",
            slots=4,
            group_col="theme",
            quotas={"t1": 2, "t2": 2},
            exit_multiplier=1.5,
        ),
        rank_sleeve=RankBufferedSleeveSpec(
            name="B",
            score_col="score_b",
            slots=2,
            group_col="industry",
            max_per_group=1,
            exit_rank=3,
            entry_rank=2,
            max_replacements=1,
        ),
        overlap_policy="aggregate",
        normal_slot_weight=0.1,
        max_name_weight=0.2,
    )


def test_build_sleeve_positions_preserves_quota_rank_buffer_overlap_and_weights() -> None:
    positions = build_sleeve_positions(_signals(), spec=_spec())

    day1 = positions.loc[positions["rebalance_date"].eq("20260105")].set_index("symbol")
    assert set(day1.index) == {"000001", "000002", "000003", "000004", "000005"}
    assert day1.loc["000002", "leg"] == "A+B"
    assert day1.loc["000002", "weight"] == 0.2
    assert day1.loc["000003", "leg"] == "B"
    assert day1["weight"].sum() == 0.6

    day2 = positions.loc[positions["rebalance_date"].eq("20260106")].set_index("symbol")
    assert set(day2.index) == {"000001", "000002", "000004", "000005", "000007"}
    assert day2.loc["000007", "leg"] == "A+B"
    assert day2.loc["000002", "leg"] == "B"
    assert day2["weight"].sum() == 0.6


def test_position_change_and_exposure_helpers_are_portfolio_owned() -> None:
    positions = build_sleeve_positions(_signals(), spec=_spec())

    changes = compute_position_changes(positions)
    exposure = compute_position_exposure(positions)

    day2_changes = changes.loc[changes["rebalance_date"].eq("20260106")]
    assert set(day2_changes["action"]) <= {"new", "exit", "weight_change", "stay"}
    assert exposure["rebalance_date"].tolist() == ["20260105", "20260106"]
    assert {"total_stocks", "total_weight", "max_industry_pct"} <= set(exposure.columns)


def test_portfolio_module_contains_no_style_replica_identity() -> None:
    spec = _spec()
    rendered = repr(spec)
    assert "StyleReplica" not in rendered
    assert "A80B20" not in rendered
