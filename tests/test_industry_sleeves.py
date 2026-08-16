from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_backtester.industry_sleeves import (
    SelectionSpec,
    attach_entry_dates,
    build_targets,
    combine_targets,
    select_industry_balanced,
    target_turnover,
    validate_targets,
)


def _formation() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["A1", "A2", "A3", "A4", "B1", "B2"],
            "industry_l1": ["A", "A", "A", "A", "B", "B"],
            "score": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        }
    )


def test_fixed_width_selection_matches_universe_industry_weights() -> None:
    result = select_industry_balanced(_formation(), score_col="score", spec=SelectionSpec(top_k=3))

    assert set(result["symbol"]) == {"A1", "A2", "B1"}
    weights = result.groupby("industry_l1")["weight"].sum()
    assert weights["A"] == pytest.approx(4 / 6)
    assert weights["B"] == pytest.approx(2 / 6)
    assert result["weight"].sum() == pytest.approx(1.0)


def test_fraction_selection_takes_each_industry_tail() -> None:
    result = select_industry_balanced(
        _formation(), score_col="score", spec=SelectionSpec(top_fraction=0.5)
    )

    assert set(result["symbol"]) == {"A1", "A2", "B1"}


def test_top_k_smaller_than_industry_count_skips_zero_seat_groups() -> None:
    formation = pd.DataFrame(
        {
            "symbol": ["A", "B", "C"],
            "industry_l1": ["A", "B", "C"],
            "score": [3.0, 2.0, 1.0],
        }
    )

    result = select_industry_balanced(formation, score_col="score", spec=SelectionSpec(top_k=2))

    assert len(result) == 2
    assert result["weight"].sum() == pytest.approx(1.0)


def test_build_and_combine_sleeves_accumulates_overlaps() -> None:
    first = _formation().assign(trade_date=pd.Timestamp("2024-01-31"))
    second = first.assign(score=-first["score"])
    sleeve_a = build_targets(first, score_col="score", spec=SelectionSpec(top_k=3))
    sleeve_b = build_targets(second, score_col="score", spec=SelectionSpec(top_k=3))

    combined = combine_targets({"a": sleeve_a, "b": sleeve_b}, {"a": 0.6, "b": 0.4})

    assert not combined.duplicated(["rebalance_date", "symbol"]).any()
    assert combined.groupby("rebalance_date")["weight"].sum().iloc[0] == pytest.approx(1.0)
    assert set(combined["symbol"]) == {"A1", "A2", "A3", "A4", "B1", "B2"}


def test_dynamic_allocations_are_applied_by_date() -> None:
    targets = pd.DataFrame(
        {
            "rebalance_date": pd.to_datetime(["2024-01-31", "2024-02-29"]),
            "symbol": ["A", "A"],
            "weight": [1.0, 1.0],
        }
    )
    other = targets.assign(symbol="B")
    allocations = pd.DataFrame(
        {
            "rebalance_date": pd.to_datetime(
                ["2024-01-31", "2024-01-31", "2024-02-29", "2024-02-29"]
            ),
            "sleeve": ["a", "b", "a", "b"],
            "allocation": [0.7, 0.3, 0.2, 0.8],
        }
    )

    result = combine_targets({"a": targets, "b": other}, allocations)

    weights = result.pivot(index="rebalance_date", columns="symbol", values="weight")
    assert weights.loc[pd.Timestamp("2024-01-31"), "A"] == pytest.approx(0.7)
    assert weights.loc[pd.Timestamp("2024-02-29"), "B"] == pytest.approx(0.8)


def test_entry_date_is_strictly_after_formation() -> None:
    targets = pd.DataFrame(
        {
            "rebalance_date": pd.to_datetime(["2024-01-31", "2024-02-02"]),
            "symbol": ["A", "A"],
            "weight": [1.0, 1.0],
        }
    )
    calendar = pd.to_datetime(["2024-01-31", "2024-02-01", "2024-02-02"])

    result = attach_entry_dates(targets, calendar)

    assert result["rebalance_date"].tolist() == [pd.Timestamp("2024-01-31")]
    assert result["entry_date"].tolist() == [pd.Timestamp("2024-02-01")]


def test_target_turnover_handles_entry_and_replacement() -> None:
    targets = pd.DataFrame(
        {
            "rebalance_date": pd.to_datetime(
                ["2024-01-31", "2024-01-31", "2024-02-29", "2024-02-29"]
            ),
            "symbol": ["A", "B", "A", "C"],
            "weight": [0.5, 0.5, 0.5, 0.5],
        }
    )

    result = target_turnover(targets)

    assert result.iloc[0] == pytest.approx(1.0)
    assert result.iloc[1] == pytest.approx(0.5)


def test_invalid_target_contract_is_rejected() -> None:
    invalid = pd.DataFrame(
        {
            "rebalance_date": [pd.Timestamp("2024-01-31")] * 2,
            "symbol": ["A", "A"],
            "weight": [0.5, 0.5],
        }
    )

    with pytest.raises(ValueError, match="duplicate"):
        validate_targets(invalid)


def test_selection_spec_requires_one_width() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        SelectionSpec()
    with pytest.raises(ValueError, match="exactly one"):
        SelectionSpec(top_k=10, top_fraction=0.2)
    with pytest.raises(ValueError, match="top_fraction"):
        SelectionSpec(top_fraction=np.nan)
