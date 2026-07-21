from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_backtester.incumbent_requalification import (
    IncumbentRequalificationPolicy,
    select_incumbent_requalified_portfolio,
)


def _cross_section(size: int = 60) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": pd.Timestamp("2026-07-20"),
            "symbol": [f"S{index:03d}" for index in range(size)],
            "selection_score": np.arange(size, 0, -1, dtype=float),
            "industry": [f"industry_{index % 10}" for index in range(size)],
            "hard_eligible": True,
            "entry_eligible": [index < 30 for index in range(size)],
        }
    )


def test_bootstrap_fills_the_portfolio_from_entry_eligible_names() -> None:
    result = select_incumbent_requalified_portfolio(_cross_section())

    assert result.positions["symbol"].tolist() == [f"S{index:03d}" for index in range(20)]
    assert result.positions["new_position"].all()
    assert result.receipt.summary["bootstrap"] is True
    assert result.receipt.summary["new_position_count"] == 20
    assert result.receipt.summary["cash_weight"] == pytest.approx(0.0)


def test_incumbents_can_survive_outside_the_entry_pool_after_current_rescoring() -> None:
    frame = _cross_section()
    frame.loc[20:24, "entry_eligible"] = False
    previous = [f"S{index:03d}" for index in range(5, 25)]

    result = select_incumbent_requalified_portfolio(frame, previous_symbols=previous)

    selected = set(result.positions["symbol"])
    assert "S020" in selected
    buffered = result.positions.loc[result.positions["buffered_incumbent"]]
    assert not buffered.empty
    assert not buffered["entry_eligible"].all()
    assert result.receipt.summary["new_position_count"] <= 4


def test_new_positions_never_bypass_entry_eligibility() -> None:
    frame = _cross_section()
    frame.loc[:9, "entry_eligible"] = False
    previous = [f"S{index:03d}" for index in range(10, 30)]

    result = select_incumbent_requalified_portfolio(frame, previous_symbols=previous)

    opened = result.positions.loc[result.positions["new_position"]]
    assert opened["entry_eligible"].all()
    assert set(opened["symbol"]).isdisjoint({f"S{index:03d}" for index in range(10)})


def test_hard_exit_and_replacement_budget_leave_cash_instead_of_forcing_weak_names() -> None:
    frame = _cross_section()
    previous = [f"S{index:03d}" for index in range(20, 40)]
    frame.loc[frame["symbol"].isin(previous[:10]), "hard_eligible"] = False
    policy = IncumbentRequalificationPolicy(max_new_positions=4)

    result = select_incumbent_requalified_portfolio(
        frame,
        previous_symbols=previous,
        policy=policy,
    )

    assert result.receipt.summary["new_position_count"] == 4
    assert len(result.positions) < 20
    assert result.receipt.summary["cash_weight"] > 0
    assert result.positions["target_weight"].eq(0.05).all()


def test_score_improvement_gate_can_reject_marginal_replacements() -> None:
    frame = _cross_section()
    previous = [f"S{index:03d}" for index in range(20)]
    frame.loc[frame["symbol"].eq("S020"), "selection_score"] = 41.1
    policy = IncumbentRequalificationPolicy(min_score_improvement=2.0)

    result = select_incumbent_requalified_portfolio(
        frame,
        previous_symbols=previous,
        policy=policy,
    )

    assert "S020" not in set(result.positions["symbol"])
    assert result.receipt.summary["score_margin_skipped_count"] >= 1


def test_industry_cap_applies_to_retained_and_new_positions() -> None:
    frame = _cross_section()
    frame.loc[:15, "industry"] = "crowded"
    policy = IncumbentRequalificationPolicy(industry_cap=3)

    result = select_incumbent_requalified_portfolio(frame, policy=policy)

    assert result.positions["industry"].value_counts().max() <= 3
    assert result.receipt.summary["industry_counts"]["crowded"] == 3


def test_policy_identity_is_deterministic_and_parameter_sensitive() -> None:
    first = IncumbentRequalificationPolicy()
    second = IncumbentRequalificationPolicy()
    changed = IncumbentRequalificationPolicy(exit_rank_limit=50)

    assert first.policy_id == second.policy_id
    assert first.policy_id != changed.policy_id


def test_invalid_hard_eligible_score_fails_closed() -> None:
    frame = _cross_section()
    frame.loc[0, "selection_score"] = np.nan

    with pytest.raises(ValueError, match="finite scores"):
        select_incumbent_requalified_portfolio(frame)


def test_empty_input_fails_closed() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        select_incumbent_requalified_portfolio(pd.DataFrame())


def test_duplicate_symbols_fail_closed() -> None:
    frame = _cross_section()
    frame.loc[1, "symbol"] = frame.loc[0, "symbol"]

    with pytest.raises(ValueError, match="one row per symbol"):
        select_incumbent_requalified_portfolio(frame)
