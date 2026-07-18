from __future__ import annotations

import pandas as pd
import pytest

from portfolio_backtester.engine import backtest_topk
from portfolio_backtester.portfolio import build_positions_by_rebalance


def _frame(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trade_date": pd.Timestamp(date),
                "symbol": symbol,
                "score": score,
                "close": 10.0,
            }
            for date, symbol, score in rows
        ]
    )


def _disjoint_candidate_frames() -> tuple[pd.DataFrame, pd.DataFrame, list[pd.Timestamp]]:
    dates = list(pd.to_datetime(["2024-01-02", "2024-01-05", "2024-01-10"]))
    old = [f"O{i}" for i in range(10)]
    new = [f"N{i}" for i in range(10)]
    score_rows = [
        {"trade_date": dates[0], "symbol": symbol, "score": 100.0 - index}
        for index, symbol in enumerate(old)
    ]
    score_rows.extend(
        {
            "trade_date": date,
            "symbol": symbol,
            "score": 100.0 - index,
        }
        for date in dates[1:]
        for index, symbol in enumerate(new)
    )
    pricing = pd.DataFrame(
        [
            {"trade_date": date, "symbol": symbol, "close": 10.0}
            for date in dates
            for symbol in [*old, *new]
        ]
    )
    scored = pd.DataFrame(score_rows).merge(pricing, on=["trade_date", "symbol"], how="left")
    return scored, pricing, dates


def test_max_new_carry_keeps_top10_discrete_and_reports_two_replacements() -> None:
    scored, pricing, dates = _disjoint_candidate_frames()

    positions = build_positions_by_rebalance(
        scored,
        pred_col="score",
        price_col="close",
        rebalance_dates=dates,
        top_k=10,
        shift_days=0,
        pricing_data=pricing,
        max_new_names_per_rebalance=2,
        max_new_names_shortfall_policy="carry",
        max_positive_names=10,
    )
    position_counts = positions.groupby("rebalance_date")["symbol"].nunique()
    assert position_counts.tolist() == [10, 10, 10]
    second = positions.loc[positions["rebalance_date"].eq("20240105")]
    assert int(second["symbol"].str.startswith("N").sum()) == 2
    assert int(second["weight"].gt(0).sum()) == 10
    assert second["weight"].sum() == pytest.approx(1.0)

    result = backtest_topk(
        scored,
        pred_col="score",
        price_col="close",
        rebalance_dates=dates,
        top_k=10,
        shift_days=0,
        cost_bps=0.0,
        trading_days_per_year=252,
        pricing_data=pricing,
        max_new_names_per_rebalance=2,
        max_new_names_shortfall_policy="carry",
        max_positive_names=10,
    )

    assert result is not None
    stats, _net, _gross, turnover, periods = result
    replacement = periods[1]
    assert turnover.iloc[1] == pytest.approx(0.2)
    assert replacement["target_name_turnover"] == pytest.approx(0.2)
    assert replacement["target_entered_names"] == ("N0", "N1")
    assert replacement["target_exited_names"] == ("O8", "O9")
    assert replacement["target_overlap_names"] == tuple(f"O{i}" for i in range(8))
    assert replacement["target_entered_count"] == 2
    assert replacement["target_exited_count"] == 2
    assert replacement["target_overlap_count"] == 8
    assert replacement["target_weight_full_l1"] == pytest.approx(0.4)
    assert replacement["target_weight_half_l1"] == pytest.approx(0.2)
    assert replacement["pretrade_demand_full_l1"] == pytest.approx(0.4)
    assert replacement["pretrade_demand_half_l1"] == pytest.approx(0.2)
    assert replacement["executed_gross"] is None
    assert replacement["executed_full_l1"] is None
    assert replacement["executed_half_l1"] is None
    assert replacement["execution_data_available"] is False
    assert periods[0]["is_initial_build"] is True
    assert periods[1]["is_initial_build"] is False
    assert stats["execution_data_available"] is False
    assert stats["initial_build_periods"] == 1
    assert stats["avg_target_weight_full_l1"] == pytest.approx(0.7)
    assert stats["avg_rebalance_target_weight_full_l1"] == pytest.approx(0.4)
    assert stats["avg_rebalance_target_entered_count"] == pytest.approx(2.0)
    assert stats["avg_rebalance_target_exited_count"] == pytest.approx(2.0)
    assert stats["avg_rebalance_target_overlap_count"] == pytest.approx(8.0)


def test_max_new_fail_policy_rejects_underfilled_target() -> None:
    scored, pricing, dates = _disjoint_candidate_frames()

    with pytest.raises(ValueError, match="underfilled the target"):
        backtest_topk(
            scored,
            pred_col="score",
            price_col="close",
            rebalance_dates=dates,
            top_k=10,
            shift_days=0,
            cost_bps=0.0,
            trading_days_per_year=252,
            pricing_data=pricing,
            max_new_names_per_rebalance=2,
            max_new_names_shortfall_policy="fail",
        )


def test_max_positive_names_rejects_turnover_interpolation_long_tail() -> None:
    scored, pricing, dates = _disjoint_candidate_frames()

    with pytest.raises(ValueError, match="exceeds max_positive_names"):
        backtest_topk(
            scored,
            pred_col="score",
            price_col="close",
            rebalance_dates=dates,
            top_k=10,
            shift_days=0,
            cost_bps=0.0,
            trading_days_per_year=252,
            pricing_data=pricing,
            max_new_names_per_rebalance=2,
            max_turnover_per_rebalance=0.4,
            max_positive_names=10,
        )


def test_selection_margin_can_use_relevance_while_ranking_numeric_score() -> None:
    dates = list(pd.to_datetime(["2024-01-02", "2024-01-05"]))
    data = pd.DataFrame(
        [
            {"trade_date": dates[0], "symbol": "A", "score": 10.0, "relevance": 0.5},
            {"trade_date": dates[0], "symbol": "B", "score": 9.0, "relevance": 0.4},
            {"trade_date": dates[1], "symbol": "B", "score": 100.0, "relevance": 0.505},
            {"trade_date": dates[1], "symbol": "A", "score": 1.0, "relevance": 0.5},
        ]
    )
    data["close"] = 10.0

    positions = build_positions_by_rebalance(
        data,
        pred_col="score",
        price_col="close",
        rebalance_dates=dates,
        top_k=1,
        shift_days=0,
        selection_score_margin=0.01,
        selection_score_margin_col="relevance",
        selection_score_margin_rank_limit=2,
    )
    numeric_margin_positions = build_positions_by_rebalance(
        data,
        pred_col="score",
        price_col="close",
        rebalance_dates=dates,
        top_k=1,
        shift_days=0,
        selection_score_margin=0.01,
        selection_score_margin_rank_limit=2,
    )

    assert positions.groupby("rebalance_date")["symbol"].first().tolist() == ["A", "A"]
    assert numeric_margin_positions.groupby("rebalance_date")["symbol"].first().tolist() == [
        "A",
        "B",
    ]


def test_selection_margin_column_missing_fails_fast() -> None:
    date = pd.Timestamp("2024-01-02")
    data = _frame([("2024-01-02", "A", 1.0)])

    with pytest.raises(ValueError, match="margin column not found"):
        build_positions_by_rebalance(
            data,
            pred_col="score",
            price_col="close",
            rebalance_dates=[date],
            top_k=1,
            shift_days=0,
            entry_dates_by_rebalance={date: date},
            selection_score_margin=0.01,
            selection_score_margin_col="relevance",
        )
