from __future__ import annotations

import pandas as pd
import pytest

from portfolio_backtester.backtest_contracts import (
    BACKTEST_PERIOD_COLUMNS,
    BACKTEST_PERIODS_CONTRACT,
    BACKTEST_RETURN_CONTRACT,
    DEFAULT_TRADABLE_FLAG_COLUMNS,
    TRADABLE_FLAGS_CONTRACT,
    assert_backtest_periods_frame,
    assert_backtest_return_frame,
    assert_tradable_flags_frame,
    build_backtest_periods_frame,
    build_backtest_return_frame,
    validate_backtest_periods_frame,
    validate_backtest_return_frame,
    validate_tradable_flags_frame,
)


def test_tradable_flags_contract_validates_boolean_columns() -> None:
    frame = pd.DataFrame(
        {
            "is_tradable": [True, False],
            "is_buy_tradable": [True, True],
            "is_sell_tradable": [True, False],
        }
    )

    assert TRADABLE_FLAGS_CONTRACT.default_columns == DEFAULT_TRADABLE_FLAG_COLUMNS
    assert validate_tradable_flags_frame(frame) == []

    invalid = frame.assign(is_tradable=[1, 0])
    assert validate_tradable_flags_frame(invalid, columns=("is_tradable",)) == [
        "is_tradable must be boolean typed"
    ]
    with pytest.raises(ValueError, match="Invalid tradable flags frame"):
        assert_tradable_flags_frame(invalid, columns=("is_tradable",))


def test_backtest_return_contract_normalizes_and_validates_period_end() -> None:
    series = pd.Series(
        [0.01, -0.02],
        index=pd.to_datetime(["2026-01-31", "2026-02-28"]),
        name="net_return",
    )

    frame = build_backtest_return_frame(series, value_column="net_return")

    assert BACKTEST_RETURN_CONTRACT.value_columns == (
        "net_return",
        "gross_return",
        "turnover",
    )
    assert frame.columns.tolist() == ["period_end", "net_return"]
    assert validate_backtest_return_frame(frame, value_column="net_return") == []

    invalid = pd.DataFrame({"period_end": ["not-a-date"], "net_return": ["bad"]})
    with pytest.raises(ValueError, match="Invalid backtest return frame"):
        assert_backtest_return_frame(invalid, value_column="net_return")


def test_backtest_periods_contract_builds_and_validates_required_columns() -> None:
    periods = [
        {
            "rebalance_date": pd.Timestamp("2026-01-05"),
            "entry_idx": 0,
            "planned_exit_idx": 1,
            "exit_idx": 2,
            "entry_date": pd.Timestamp("2026-01-06"),
            "planned_exit_date": pd.Timestamp("2026-01-07"),
            "exit_date": pd.Timestamp("2026-01-08"),
            "exit_delay_steps": 1,
        }
    ]

    frame = build_backtest_periods_frame(periods)

    assert BACKTEST_PERIODS_CONTRACT.required_columns == BACKTEST_PERIOD_COLUMNS
    assert frame.columns.tolist() == list(BACKTEST_PERIOD_COLUMNS)
    assert validate_backtest_periods_frame(frame) == []

    invalid = frame.drop(columns=["exit_date"])
    with pytest.raises(ValueError, match="Invalid backtest periods frame"):
        assert_backtest_periods_frame(invalid)
