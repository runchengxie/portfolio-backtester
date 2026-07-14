from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_backtester.daily_watch20 import (
    DailyWatch20Config,
    DailyWatch20SelectionError,
    GuardFactorSpec,
    select_daily_watch20,
)


def _cross_section(
    *,
    size: int = 60,
    date: str = "2026-07-10",
    reverse_scores: bool = False,
) -> pd.DataFrame:
    score = np.arange(size, 0, -1, dtype=float)
    if reverse_scores:
        score = score[::-1]
    return pd.DataFrame(
        {
            "trade_date": pd.Timestamp(date),
            "symbol": [f"S{idx:03d}" for idx in range(size)],
            "first_industry_name": [f"industry_{idx % 10}" for idx in range(size)],
            "xgb_score": score,
            "guard_score": score,
            "hard_eligible": True,
        }
    )


def _assert_strict_watch20(result) -> None:
    watchlist = result.watchlist
    assert len(watchlist) == 20
    assert watchlist["symbol"].nunique() == 20
    assert watchlist["tracking_weight"].sum() == pytest.approx(1.0)
    assert watchlist["first_industry_name"].value_counts().max() <= 4
    assert result.receipt.summary["selected_count"] == 20
    assert result.receipt.summary["unique_symbol_count"] == 20
    assert result.receipt.summary["tracking_weight_sum"] == pytest.approx(1.0)


def test_a_is_pure_ml_and_b_uses_default_sixty_forty_prior() -> None:
    frame = _cross_section()
    frame.loc[:3, "guard_score"] = np.nan

    result = select_daily_watch20(frame)

    _assert_strict_watch20(result)
    watchlist = result.watchlist
    selected_a = watchlist.loc[watchlist["sleeve"].eq("A")]
    selected_b = watchlist.loc[watchlist["sleeve"].eq("B")]
    assert selected_a["symbol"].tolist() == ["S000", "S001", "S002", "S003"]
    assert selected_a["guard_prior"].isna().all()
    assert set(selected_a["symbol"]).isdisjoint(selected_b["symbol"])
    assert np.allclose(
        selected_b["blended_score"],
        0.60 * selected_b["ml_percentile"] + 0.40 * selected_b["guard_prior"],
    )
    assert result.receipt.status == "selected"
    assert result.receipt.summary["a_selected_count"] == 4
    assert result.receipt.summary["b_selected_count"] == 16


def test_guard_prior_supports_weighted_factors_and_dual_confirmation() -> None:
    frame = _cross_section()
    frame["quality"] = frame["guard_score"]
    frame["risk"] = np.arange(len(frame), dtype=float)
    config = DailyWatch20Config(
        guard_factors=(
            GuardFactorSpec("quality", weight=2.0),
            GuardFactorSpec("risk", weight=1.0, higher_is_better=False),
        )
    )

    result = select_daily_watch20(frame, config=config)

    _assert_strict_watch20(result)
    eligible = frame.copy()
    expected_quality = eligible["quality"].rank(method="average", pct=True)
    expected_risk = eligible["risk"].rank(method="average", ascending=False, pct=True)
    expected_guard = dict(
        zip(eligible["symbol"], (2.0 * expected_quality + expected_risk) / 3.0, strict=False)
    )
    selected_b = result.watchlist.loc[result.watchlist["sleeve"].eq("B")]
    assert selected_b["guard_prior"].tolist() == pytest.approx(
        [expected_guard[symbol] for symbol in selected_b["symbol"]]
    )
    selected_a = result.watchlist.loc[result.watchlist["sleeve"].eq("A")]
    assert selected_a["dual_confirmed"].all()
    assert result.receipt.summary["dual_confirmed_count"] == 4


def test_global_industry_cap_refills_to_twenty_unique_names() -> None:
    frame = _cross_section()
    frame.loc[:11, "first_industry_name"] = "crowded"

    result = select_daily_watch20(frame)

    _assert_strict_watch20(result)
    counts = result.watchlist["first_industry_name"].value_counts()
    assert counts["crowded"] == 4
    assert max(result.receipt.summary["industry_counts"].values()) == 4


def test_a_is_not_re_ranked_by_industry_and_fails_closed_on_global_cap() -> None:
    frame = _cross_section()
    frame.loc[:3, "first_industry_name"] = "crowded"

    with pytest.raises(DailyWatch20SelectionError) as exc_info:
        select_daily_watch20(frame, config=DailyWatch20Config(industry_cap=3))

    receipt = exc_info.value.receipt
    assert receipt.status == "unavailable"
    assert receipt.reason == "pure-ML A selection exceeds the global industry cap"
    assert receipt.summary["a_industry_counts"] == {"crowded": 4}


def test_b_retention_and_replacement_budget_limit_discretionary_churn() -> None:
    first = select_daily_watch20(_cross_section(date="2026-07-09"))
    previous_b = first.watchlist.loc[first.watchlist["sleeve"].eq("B"), "symbol"].tolist()

    second = select_daily_watch20(
        _cross_section(date="2026-07-10", reverse_scores=True),
        previous_b_symbols=previous_b,
    )

    _assert_strict_watch20(second)
    selected_b = second.watchlist.loc[second.watchlist["sleeve"].eq("B")]
    assert int(selected_b["retained_b"].sum()) >= 12
    assert second.receipt.summary["b_replacement_count"] <= 4
    assert second.receipt.summary["b_replacement_limit_forced"] is False


def test_hard_ineligibility_can_force_replacements_beyond_budget() -> None:
    first = select_daily_watch20(_cross_section(date="2026-07-09"))
    previous_b = first.watchlist.loc[first.watchlist["sleeve"].eq("B"), "symbol"].tolist()
    next_frame = _cross_section(date="2026-07-10")
    next_frame.loc[next_frame["symbol"].isin(previous_b[:6]), "hard_eligible"] = False

    second = select_daily_watch20(next_frame, previous_b_symbols=previous_b)

    _assert_strict_watch20(second)
    assert second.receipt.summary["b_replacement_count"] == 6
    assert second.receipt.summary["b_forced_replacement_count"] == 2
    assert second.receipt.summary["b_replacement_limit_forced"] is True
    assert set(previous_b[:6]).isdisjoint(second.watchlist["symbol"])


def test_explicit_core20_fallback_has_no_exploration_sleeve() -> None:
    result = select_daily_watch20(
        _cross_section(),
        fallback_mode="core20",
        fallback_reason="exploration_model_not_promotable",
    )

    _assert_strict_watch20(result)
    assert set(result.watchlist["sleeve"]) == {"B"}
    assert not result.watchlist["dual_confirmed"].any()
    assert result.watchlist["tracking_weight"].tolist() == pytest.approx([0.05] * 20)
    assert result.receipt.status == "fallback"
    assert result.receipt.fallback_mode == "core20"
    assert result.receipt.reason == "exploration_model_not_promotable"
    assert result.receipt.summary["a_selected_count"] == 0
    assert result.receipt.summary["b_selected_count"] == 20


def test_insufficient_candidates_fail_closed_with_receipt() -> None:
    with pytest.raises(DailyWatch20SelectionError) as exc_info:
        select_daily_watch20(_cross_section(size=12))

    receipt = exc_info.value.receipt
    assert receipt.status == "unavailable"
    assert receipt.fallback_mode == "none"
    assert receipt.reason == "insufficient B candidates after deduplication and industry cap"
    assert receipt.summary["a_selected_count"] == 4
    assert receipt.summary["b_selected_count"] == 8
    assert receipt.summary["hard_eligible_count"] == 12
