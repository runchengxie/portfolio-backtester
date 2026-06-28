from __future__ import annotations

import pandas as pd
import pytest

from cstree.backtesting.contracts import (
    BACKTEST_PRICING_CONTRACT,
    BACKTEST_PRICING_CONTRACT_NAME,
    BACKTEST_PRICING_KEY_COLUMNS,
    CANONICAL_POSITIONS_BY_REBALANCE_FILE,
    POSITIONS_BY_REBALANCE_CONTRACT,
    POSITIONS_BY_REBALANCE_CONTRACT_NAME,
    POSITIONS_BY_REBALANCE_REQUIRED_COLUMNS,
    STRATEGY_SPEC_CONTRACT,
    STRATEGY_SPEC_REQUIRED_FIELDS,
    GroupCap,
    StrategySpec,
    assert_backtest_pricing_frame,
    assert_positions_by_rebalance_frame,
    assert_strategy_spec,
    required_backtest_pricing_columns,
    validate_backtest_pricing_frame,
    validate_positions_by_rebalance_frame,
    validate_strategy_spec,
)


def _pricing_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2026-01-05", "2026-01-06"]),
            "symbol": ["600519.SH", "600519.SH"],
            "open": [100.0, 101.0],
            "close": [101.0, 102.0],
            "amount": [10_000.0, 12_000.0],
            "is_tradable": [True, False],
        }
    )


def _positions_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rebalance_date": ["20260105", "20260105"],
            "entry_date": ["20260106", "20260106"],
            "symbol": ["600519.SH", "000001.SZ"],
            "weight": [0.6, 0.4],
            "signal": [1.2, 0.8],
            "rank": [1, 2],
            "side": ["long", "long"],
        }
    )


def test_backtest_pricing_contract_validates_required_columns() -> None:
    required = required_backtest_pricing_columns(
        entry_price_col="open",
        exit_price_col="close",
        amount_columns=("amount", "amount"),
        tradable_col="is_tradable",
    )
    issues = validate_backtest_pricing_frame(
        _pricing_frame(),
        entry_price_col="open",
        exit_price_col="close",
        amount_columns=("amount",),
        tradable_col="is_tradable",
        require_two_trade_dates=True,
    )

    assert BACKTEST_PRICING_CONTRACT.name == BACKTEST_PRICING_CONTRACT_NAME
    assert BACKTEST_PRICING_CONTRACT.key_columns == BACKTEST_PRICING_KEY_COLUMNS
    assert required == ("trade_date", "symbol", "open", "close", "amount", "is_tradable")
    assert issues == []


def test_backtest_pricing_contract_reports_invalid_frame() -> None:
    missing = _pricing_frame().drop(columns=["amount"])

    issues = validate_backtest_pricing_frame(
        missing,
        entry_price_col="open",
        exit_price_col="close",
        amount_columns=("amount",),
        tradable_col="is_tradable",
    )

    assert issues == ["missing columns: amount"]
    with pytest.raises(ValueError, match="Invalid backtest pricing frame"):
        assert_backtest_pricing_frame(
            missing,
            entry_price_col="open",
            exit_price_col="close",
            amount_columns=("amount",),
            tradable_col="is_tradable",
        )


def test_positions_by_rebalance_contract_validates_artifact_shape() -> None:
    issues = validate_positions_by_rebalance_frame(_positions_frame())

    assert POSITIONS_BY_REBALANCE_CONTRACT.name == POSITIONS_BY_REBALANCE_CONTRACT_NAME
    assert POSITIONS_BY_REBALANCE_CONTRACT.file_name == CANONICAL_POSITIONS_BY_REBALANCE_FILE
    assert (
        POSITIONS_BY_REBALANCE_CONTRACT.required_columns == POSITIONS_BY_REBALANCE_REQUIRED_COLUMNS
    )
    assert issues == []


def test_positions_by_rebalance_contract_reports_invalid_frame() -> None:
    missing = _positions_frame().drop(columns=["weight"])
    invalid = _positions_frame().assign(rebalance_date=["not-a-date", "20260105"], weight=["x", 1])

    assert validate_positions_by_rebalance_frame(missing) == ["missing columns: weight"]
    assert validate_positions_by_rebalance_frame(invalid) == [
        "rebalance_date must be date-like",
        "weight must be numeric",
    ]
    with pytest.raises(ValueError, match="Invalid positions_by_rebalance frame"):
        assert_positions_by_rebalance_frame(missing)


def test_strategy_contract_validates_and_serializes_spec() -> None:
    spec = StrategySpec(
        name="topk-demo",
        type="topk_buffered_long_only",
        score_col="signal_backtest",
        top_k=20,
        buffer_exit=5,
        buffer_entry=2,
        group_cap=GroupCap(column="industry", max_names=3),
    )

    assert STRATEGY_SPEC_CONTRACT.required_fields == STRATEGY_SPEC_REQUIRED_FIELDS
    assert validate_strategy_spec(spec) == []
    assert spec.to_dict()["group_cap"] == {"column": "industry", "max_names": 3}


def test_strategy_contract_reports_invalid_spec() -> None:
    invalid = StrategySpec(
        name="",
        type="topk_buffered_long_only",
        score_col="",
        top_k=0,
        buffer_exit=-1,
        group_cap=GroupCap(column="", max_names=0),
    )

    issues = validate_strategy_spec(invalid)

    assert "name must be non-empty" in issues
    assert "score_col must be non-empty" in issues
    assert "top_k must be > 0" in issues
    assert "buffer_exit must be >= 0" in issues
    assert "group_cap.max_names must be > 0" in issues
    with pytest.raises(ValueError, match="Invalid strategy spec"):
        assert_strategy_spec(invalid)
