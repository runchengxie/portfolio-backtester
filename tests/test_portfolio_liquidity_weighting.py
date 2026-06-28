from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cstree.backtesting.portfolio import build_position_weights, build_positions_by_rebalance


def test_sqrt_liquidity_weighting_uses_liquidity_and_caps_concentration():
    symbols = [f"S{i:03d}" for i in range(100)]
    day = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-01"] * 100),
            "symbol": symbols,
            "score": list(range(100, 0, -1)),
            "medadv20_amount": np.geomspace(100.0, 10_000.0, 100),
        }
    )

    weights = build_position_weights(
        day,
        symbols,
        "score",
        side="long",
        weighting="sqrt_liquidity",
    )

    assert weights.sum() == pytest.approx(1.0)
    assert weights.iloc[-10:].mean() > weights.iloc[:10].mean()
    assert float(weights.max()) <= 0.05 + 1e-12
    assert weights.iloc[0] != pytest.approx(1.0 / 100)


def test_build_positions_can_apply_quantile_liquidity_floor_before_topk():
    trade_date = pd.Timestamp("2024-01-01")
    entry_date = pd.Timestamp("2024-01-02")
    data = pd.DataFrame(
        {
            "trade_date": [trade_date] * 4,
            "symbol": ["LOW", "MID", "HIGH", "TOP"],
            "score": [100.0, 90.0, 80.0, 70.0],
            "close": [10.0, 10.0, 10.0, 10.0],
            "medadv20_amount": [1.0, 2.0, 3.0, 4.0],
        }
    )
    pricing = pd.DataFrame(
        {
            "trade_date": [trade_date] * 4 + [entry_date] * 4,
            "symbol": ["LOW", "MID", "HIGH", "TOP"] * 2,
            "close": [10.0] * 8,
            "medadv20_amount": [1.0, 2.0, 3.0, 4.0] * 2,
        }
    )

    positions = build_positions_by_rebalance(
        data,
        pred_col="score",
        price_col="close",
        rebalance_dates=[trade_date],
        top_k=2,
        shift_days=1,
        weighting="equal",
        pricing_data=pricing,
        liquidity_floor_col="medadv20_amount",
        liquidity_floor_quantile=0.5,
    )

    assert positions["symbol"].tolist() == ["HIGH", "TOP"]


def test_build_positions_can_tiebreak_within_score_bucket_by_size():
    trade_date = pd.Timestamp("2024-01-01")
    entry_date = pd.Timestamp("2024-01-02")
    data = pd.DataFrame(
        {
            "trade_date": [trade_date] * 3,
            "symbol": ["SMALL", "LARGE", "LOW"],
            "score": [1.00004, 1.00003, 0.5],
            "close": [10.0, 10.0, 10.0],
            "total_mv": [1.0, 100.0, 1000.0],
        }
    )
    pricing = pd.DataFrame(
        {
            "trade_date": [trade_date] * 3 + [entry_date] * 3,
            "symbol": ["SMALL", "LARGE", "LOW"] * 2,
            "close": [10.0] * 6,
        }
    )

    positions = build_positions_by_rebalance(
        data,
        pred_col="score",
        price_col="close",
        rebalance_dates=[trade_date],
        top_k=1,
        shift_days=1,
        weighting="equal",
        pricing_data=pricing,
        selection_tiebreak_col="total_mv",
        selection_score_bucket_size=0.0001,
    )

    assert positions["symbol"].tolist() == ["LARGE"]


def test_build_positions_score_margin_keeps_prior_holding_with_close_score():
    first = pd.Timestamp("2024-01-01")
    second = pd.Timestamp("2024-01-08")
    data = pd.DataFrame(
        {
            "trade_date": [first, first, second, second],
            "symbol": ["OLD", "LOW", "OLD", "NEW"],
            "score": [2.0, 1.0, 1.00000, 1.00004],
            "close": [10.0, 10.0, 10.0, 10.0],
        }
    )

    positions = build_positions_by_rebalance(
        data,
        pred_col="score",
        price_col="close",
        rebalance_dates=[first, second],
        top_k=1,
        shift_days=0,
        weighting="equal",
        selection_score_margin=0.00005,
        selection_score_margin_rank_limit=2,
    )

    assert positions.loc[positions["rebalance_date"].eq("20240108"), "symbol"].tolist() == ["OLD"]


def test_build_positions_supports_sqrt_liquidity_weighting_with_pricing_liquidity():
    trade_date = pd.Timestamp("2024-01-01")
    entry_date = pd.Timestamp("2024-01-02")
    symbols = [f"S{i:03d}" for i in range(100)]
    data = pd.DataFrame(
        {
            "trade_date": [trade_date] * 100,
            "symbol": symbols,
            "score": list(range(100, 0, -1)),
            "close": [10.0] * 100,
        }
    )
    pricing = pd.DataFrame(
        {
            "trade_date": [trade_date] * 100 + [entry_date] * 100,
            "symbol": symbols * 2,
            "close": [10.0] * 200,
            "medadv20_amount": list(np.geomspace(100.0, 10_000.0, 100)) * 2,
        }
    )

    positions = build_positions_by_rebalance(
        data,
        pred_col="score",
        price_col="close",
        rebalance_dates=[trade_date],
        top_k=100,
        shift_days=1,
        weighting="sqrt_liquidity",
        pricing_data=pricing,
        weighting_liquidity_col="medadv20_amount",
    )

    weights = positions.set_index("symbol")["weight"]
    assert weights.sum() == pytest.approx(1.0)
    assert weights.tail(10).mean() > weights.head(10).mean()
