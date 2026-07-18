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


def test_selection_min_score_leaves_long_portfolio_below_top_k() -> None:
    date = pd.Timestamp("2024-01-02")
    data = _frame(
        [
            ("2024-01-02", "A", 3.0),
            ("2024-01-02", "B", 2.0),
            ("2024-01-02", "C", 1.0),
            ("2024-01-02", "D", 0.0),
        ]
    )

    positions = build_positions_by_rebalance(
        data,
        pred_col="score",
        price_col="close",
        rebalance_dates=[date],
        top_k=3,
        shift_days=0,
        entry_dates_by_rebalance={date: date},
        selection_min_score=2.0,
    )

    assert positions["symbol"].tolist() == ["A", "B"]


def test_selection_controls_keep_empty_input_empty() -> None:
    data = pd.DataFrame(columns=["trade_date", "symbol", "score", "close"])

    positions = build_positions_by_rebalance(
        data,
        pred_col="score",
        price_col="close",
        rebalance_dates=[pd.Timestamp("2024-01-02")],
        top_k=3,
        shift_days=0,
        selection_min_score=2.0,
        max_new_names_per_rebalance=1,
    )

    assert positions.empty


def test_selection_min_score_uses_ascending_semantics_for_short_side() -> None:
    date = pd.Timestamp("2024-01-02")
    data = _frame(
        [
            ("2024-01-02", "A", 5.0),
            ("2024-01-02", "B", 4.0),
            ("2024-01-02", "C", 3.0),
            ("2024-01-02", "D", 2.0),
            ("2024-01-02", "E", 1.0),
        ]
    )

    positions = build_positions_by_rebalance(
        data,
        pred_col="score",
        price_col="close",
        rebalance_dates=[date],
        top_k=1,
        shift_days=0,
        long_only=False,
        short_k=2,
        entry_dates_by_rebalance={date: date},
        selection_min_score=1.5,
    )

    longs = positions.loc[positions["side"].eq("long"), "symbol"].tolist()
    shorts = positions.loc[positions["side"].eq("short"), "symbol"].tolist()
    assert longs == ["A"]
    assert shorts == ["E"]


def test_long_short_threshold_uses_actual_long_count_for_short_capacity() -> None:
    date = pd.Timestamp("2024-01-02")
    data = _frame(
        [
            ("2024-01-02", "A", 5.0),
            ("2024-01-02", "B", -1.0),
            ("2024-01-02", "C", -2.0),
            ("2024-01-02", "D", -3.0),
            ("2024-01-02", "E", -4.0),
        ]
    )

    positions = build_positions_by_rebalance(
        data,
        pred_col="score",
        price_col="close",
        rebalance_dates=[date],
        top_k=4,
        shift_days=0,
        long_only=False,
        short_k=4,
        entry_dates_by_rebalance={date: date},
        selection_min_score=0.0,
    )

    assert positions.loc[positions["side"].eq("long"), "symbol"].tolist() == ["A"]
    assert positions.loc[positions["side"].eq("short"), "symbol"].tolist() == [
        "E",
        "D",
        "C",
        "B",
    ]


def test_max_new_names_exempts_initial_build_and_limits_later_replacements() -> None:
    first = pd.Timestamp("2024-01-02")
    second = pd.Timestamp("2024-01-09")
    data = _frame(
        [
            ("2024-01-02", "A", 6.0),
            ("2024-01-02", "B", 5.0),
            ("2024-01-02", "C", 4.0),
            ("2024-01-02", "D", 3.0),
            ("2024-01-09", "D", 9.0),
            ("2024-01-09", "E", 8.0),
            ("2024-01-09", "F", 7.0),
            ("2024-01-09", "A", 6.0),
            ("2024-01-09", "B", 5.0),
            ("2024-01-09", "C", 4.0),
        ]
    )

    positions = build_positions_by_rebalance(
        data,
        pred_col="score",
        price_col="close",
        rebalance_dates=[first, second],
        top_k=3,
        shift_days=0,
        max_new_names_per_rebalance=1,
    )

    first_names = positions.loc[positions["rebalance_date"].eq("20240102"), "symbol"].tolist()
    second_names = positions.loc[positions["rebalance_date"].eq("20240109"), "symbol"].tolist()
    assert first_names == ["A", "B", "C"]
    assert second_names == ["D", "A", "B"]


def test_max_new_names_runs_after_tradability_and_respects_group_cap() -> None:
    first = pd.Timestamp("2024-01-02")
    second = pd.Timestamp("2024-01-09")
    rows = [
        ("2024-01-02", "A", 6.0, "X", True),
        ("2024-01-02", "B", 5.0, "Y", True),
        ("2024-01-02", "C", 4.0, "Z", True),
        ("2024-01-09", "D", 10.0, "W", False),
        ("2024-01-09", "E", 9.0, "X", True),
        ("2024-01-09", "F", 8.0, "V", True),
        ("2024-01-09", "A", 7.0, "X", True),
        ("2024-01-09", "B", 6.0, "Y", True),
        ("2024-01-09", "C", 5.0, "Z", True),
    ]
    data = pd.DataFrame(
        [
            {
                "trade_date": pd.Timestamp(date),
                "symbol": symbol,
                "score": score,
                "close": 10.0,
                "sector": sector,
                "is_tradable": tradable,
            }
            for date, symbol, score, sector, tradable in rows
        ]
    )

    positions = build_positions_by_rebalance(
        data,
        pred_col="score",
        price_col="close",
        rebalance_dates=[first, second],
        top_k=3,
        shift_days=0,
        tradable_col="is_tradable",
        group_col="sector",
        max_names_per_group=1,
        max_new_names_per_rebalance=1,
    )

    second_names = positions.loc[positions["rebalance_date"].eq("20240109"), "symbol"].tolist()
    assert second_names == ["E", "B", "C"]


def test_threshold_and_new_name_limit_do_not_refill_ineligible_names() -> None:
    first = pd.Timestamp("2024-01-02")
    second = pd.Timestamp("2024-01-09")
    data = _frame(
        [
            ("2024-01-02", "A", 6.0),
            ("2024-01-02", "B", 5.0),
            ("2024-01-02", "C", 4.0),
            ("2024-01-09", "D", 9.0),
            ("2024-01-09", "E", 8.0),
            ("2024-01-09", "F", 7.0),
            ("2024-01-09", "A", 2.0),
            ("2024-01-09", "B", 1.0),
            ("2024-01-09", "C", 0.0),
        ]
    )

    positions = build_positions_by_rebalance(
        data,
        pred_col="score",
        price_col="close",
        rebalance_dates=[first, second],
        top_k=3,
        shift_days=0,
        selection_min_score=4.0,
        max_new_names_per_rebalance=1,
    )

    second_names = positions.loc[positions["rebalance_date"].eq("20240109"), "symbol"].tolist()
    assert second_names == ["D"]


def test_score_threshold_takes_precedence_over_weight_turnover_cap() -> None:
    first = pd.Timestamp("2024-01-02")
    second = pd.Timestamp("2024-01-09")
    data = _frame(
        [
            ("2024-01-02", "A", 6.0),
            ("2024-01-02", "B", 5.0),
            ("2024-01-02", "C", 1.0),
            ("2024-01-02", "D", 0.0),
            ("2024-01-09", "C", 9.0),
            ("2024-01-09", "D", 8.0),
            ("2024-01-09", "A", 2.0),
            ("2024-01-09", "B", 1.0),
        ]
    )

    positions = build_positions_by_rebalance(
        data,
        pred_col="score",
        price_col="close",
        rebalance_dates=[first, second],
        top_k=2,
        shift_days=0,
        selection_min_score=4.0,
        max_turnover_per_rebalance=0.1,
    )

    second_names = positions.loc[positions["rebalance_date"].eq("20240109"), "symbol"].tolist()
    assert second_names == ["C", "D"]


def test_backtest_records_weak_signal_period_as_cash() -> None:
    dates = pd.to_datetime(["2024-01-02", "2024-01-09", "2024-01-16", "2024-01-23"])
    rows: list[dict[str, object]] = []
    for date, score, price in zip(
        dates,
        [10.0, 0.0, 10.0, 10.0],
        [100.0, 110.0, 110.0, 121.0],
        strict=True,
    ):
        rows.append({"trade_date": date, "symbol": "A", "score": score, "close": price})
    data = pd.DataFrame(rows)

    result = backtest_topk(
        data,
        pred_col="score",
        price_col="close",
        rebalance_dates=list(dates),
        top_k=1,
        shift_days=0,
        cost_bps=10.0,
        trading_days_per_year=52,
        selection_min_score=5.0,
    )

    assert result is not None
    stats, net, gross, turnover, periods = result
    assert stats["periods"] == 3
    assert gross.tolist() == pytest.approx([0.10, 0.0, 0.10])
    assert net.tolist() == pytest.approx([0.099, -0.001, 0.099])
    assert turnover.tolist() == pytest.approx([1.0, 0.5, 0.5])
    assert [period["rebalance_date"] for period in periods] == list(dates[:3])


def test_backtest_long_short_threshold_does_not_underfill_short_leg() -> None:
    first = pd.Timestamp("2024-01-02")
    second = pd.Timestamp("2024-01-09")
    scores = {"A": 5.0, "B": -1.0, "C": -2.0, "D": -3.0, "E": -4.0}
    ending_prices = {"A": 10.0, "B": 9.0, "C": 8.0, "D": 7.0, "E": 6.0}
    data = pd.DataFrame(
        [
            {
                "trade_date": date,
                "symbol": symbol,
                "score": score,
                "close": 10.0 if date == first else ending_prices[symbol],
            }
            for date in (first, second)
            for symbol, score in scores.items()
        ]
    )

    result = backtest_topk(
        data,
        pred_col="score",
        price_col="close",
        rebalance_dates=[first, second],
        top_k=4,
        short_k=4,
        long_only=False,
        shift_days=0,
        cost_bps=0.0,
        trading_days_per_year=52,
        selection_min_score=0.0,
    )

    assert result is not None
    _stats, _net, gross, _turnover, _periods = result
    assert gross.tolist() == pytest.approx([0.25])


def test_backtest_applies_new_name_budget_after_initial_build() -> None:
    dates = pd.to_datetime(["2024-01-02", "2024-01-09", "2024-01-16"])
    scores_by_date = {
        dates[0]: {"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0},
        dates[1]: {"A": 2.0, "B": 1.0, "C": 4.0, "D": 3.0},
        dates[2]: {"A": 2.0, "B": 1.0, "C": 4.0, "D": 3.0},
    }
    ending_prices = {"A": 10.0, "B": 10.0, "C": 11.0, "D": 20.0}
    data = pd.DataFrame(
        [
            {
                "trade_date": date,
                "symbol": symbol,
                "score": score,
                "close": ending_prices[symbol] if date == dates[2] else 10.0,
            }
            for date in dates
            for symbol, score in scores_by_date[date].items()
        ]
    )

    result = backtest_topk(
        data,
        pred_col="score",
        price_col="close",
        rebalance_dates=list(dates),
        top_k=2,
        shift_days=0,
        cost_bps=0.0,
        trading_days_per_year=52,
        max_new_names_per_rebalance=1,
    )

    assert result is not None
    _stats, _net, gross, _turnover, _periods = result
    assert gross.tolist() == pytest.approx([0.0, 0.05])


def test_first_non_empty_selection_is_treated_as_initial_build() -> None:
    first = pd.Timestamp("2024-01-02")
    second = pd.Timestamp("2024-01-09")
    data = _frame(
        [
            ("2024-01-02", "A", 1.0),
            ("2024-01-02", "B", 0.0),
            ("2024-01-09", "C", 9.0),
            ("2024-01-09", "D", 8.0),
            ("2024-01-09", "E", 7.0),
        ]
    )

    positions = build_positions_by_rebalance(
        data,
        pred_col="score",
        price_col="close",
        rebalance_dates=[first, second],
        top_k=3,
        shift_days=0,
        selection_min_score=5.0,
        max_new_names_per_rebalance=1,
    )

    assert positions["rebalance_date"].unique().tolist() == ["20240109"]
    assert positions["symbol"].tolist() == ["C", "D", "E"]


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), float("-inf")])
def test_selection_min_score_must_be_finite(threshold: float) -> None:
    date = pd.Timestamp("2024-01-02")
    data = _frame([("2024-01-02", "A", 1.0)])

    with pytest.raises(ValueError, match="selection_min_score must be finite"):
        build_positions_by_rebalance(
            data,
            pred_col="score",
            price_col="close",
            rebalance_dates=[date],
            top_k=1,
            shift_days=0,
            selection_min_score=threshold,
        )


def test_max_new_names_must_be_non_negative() -> None:
    date = pd.Timestamp("2024-01-02")
    data = _frame([("2024-01-02", "A", 1.0)])

    with pytest.raises(ValueError, match="max_new_names_per_rebalance must be >= 0"):
        build_positions_by_rebalance(
            data,
            pred_col="score",
            price_col="close",
            rebalance_dates=[date],
            top_k=1,
            shift_days=0,
            max_new_names_per_rebalance=-1,
        )


@pytest.mark.parametrize("invalid", [True, 1.5, "1"])
def test_max_new_names_must_be_an_integer(invalid: object) -> None:
    date = pd.Timestamp("2024-01-02")
    data = _frame([("2024-01-02", "A", 1.0)])

    with pytest.raises(ValueError, match="non-negative integer"):
        build_positions_by_rebalance(
            data,
            pred_col="score",
            price_col="close",
            rebalance_dates=[date],
            top_k=1,
            shift_days=0,
            max_new_names_per_rebalance=invalid,  # type: ignore[arg-type]
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
