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


@pytest.mark.parametrize("weighting", ["equal", "signal"])
def test_selection_controls_do_not_count_duplicate_symbol_twice(weighting: str) -> None:
    date = pd.Timestamp("2024-01-02")
    data = _frame(
        [
            ("2024-01-02", "A", 3.0),
            ("2024-01-02", "A", 3.0),
            ("2024-01-02", "B", 2.0),
        ]
    )

    positions = build_positions_by_rebalance(
        data,
        pred_col="score",
        price_col="close",
        rebalance_dates=[date],
        top_k=2,
        shift_days=0,
        entry_dates_by_rebalance={date: date},
        selection_min_score=2.0,
        max_new_names_per_rebalance=1,
        weighting=weighting,
    )

    assert positions["symbol"].tolist() == ["A", "B"]


def test_backtest_controls_deduplicate_before_signal_weighting() -> None:
    first = pd.Timestamp("2024-01-02")
    second = pd.Timestamp("2024-01-09")
    data = _frame(
        [
            ("2024-01-02", "A", 3.0),
            ("2024-01-02", "A", 1.0),
            ("2024-01-02", "B", 2.0),
            ("2024-01-09", "A", 3.0),
            ("2024-01-09", "B", 2.0),
        ]
    )

    result = backtest_topk(
        data,
        pred_col="score",
        price_col="close",
        rebalance_dates=[first, second],
        top_k=2,
        shift_days=0,
        cost_bps=0.0,
        trading_days_per_year=252,
        weighting="signal",
        selection_min_score=2.0,
        max_new_names_per_rebalance=1,
    )

    assert result is not None
    assert result[0]["periods"] == 1
