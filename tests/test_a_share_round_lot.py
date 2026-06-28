from __future__ import annotations

import math

import pandas as pd

from cstree.backtesting.a_share_round_lot import (
    RoundLotVariant,
    allocate_round_lot,
    allocate_round_lot_account,
    cap_and_redistribute,
    portfolio_value,
    select_round_lot_targets,
)


def test_cap_and_redistribute_respects_feasible_cap_and_normalizes():
    weights = cap_and_redistribute(pd.Series([9.0, 1.0, 1.0, 1.0]), cap=0.4)

    assert math.isclose(float(weights.sum()), 1.0)
    assert float(weights.max()) <= 0.4 + 1e-12
    assert math.isclose(float(weights.iloc[0]), 0.4)


def test_cap_and_redistribute_falls_back_when_cap_is_infeasible():
    weights = cap_and_redistribute(pd.Series([3.0, 1.0]), cap=0.4)

    assert math.isclose(float(weights.sum()), 1.0)
    assert weights.to_dict() == {0: 0.75, 1: 0.25}


def test_select_round_lot_targets_applies_liquidity_prior_holdings_and_industry_cap():
    day = pd.DataFrame(
        [
            {
                "symbol": "000001.SZ",
                "signal_backtest": 0.90,
                "medadv20_amount": 100.0,
                "first_industry_name": "Bank",
            },
            {
                "symbol": "000002.SZ",
                "signal_backtest": 0.80,
                "medadv20_amount": 90.0,
                "first_industry_name": "Bank",
            },
            {
                "symbol": "000003.SZ",
                "signal_backtest": 0.70,
                "medadv20_amount": 80.0,
                "first_industry_name": "Tech",
            },
            {
                "symbol": "000004.SZ",
                "signal_backtest": 0.10,
                "medadv20_amount": 70.0,
                "first_industry_name": "Consumer",
            },
        ]
    )
    variant = RoundLotVariant(
        target_holdings=3,
        liquidity_floor_q=0.0,
        weighting="equal",
        industry_cap=1,
        max_weight=0.4,
    )

    selected = select_round_lot_targets(day, variant, previous_symbols={"000004.SZ"})

    assert selected["symbol"].tolist() == ["000004.SZ", "000001.SZ", "000003.SZ"]
    assert selected["first_industry_name"].tolist() == ["Consumer", "Bank", "Tech"]
    assert selected["target_weight"].round(6).tolist() == [0.333333, 0.333333, 0.333333]


def test_allocate_round_lot_skips_unaffordable_and_reports_weight_error():
    targets = pd.DataFrame(
        [
            {"symbol": "cheap", "target_weight": 0.50},
            {"symbol": "expensive", "target_weight": 0.30},
            {"symbol": "missing", "target_weight": 0.20},
        ]
    )
    prices = pd.Series({"cheap": 9.0, "expensive": 2_000.0})

    allocation, diagnostics = allocate_round_lot(
        targets,
        prices,
        equity=100_000.0,
        round_lot=100,
        min_notional=5_000.0,
        max_weight=0.5,
    )

    assert allocation == {"cheap": 5_500}
    assert diagnostics["target_names"] == 3
    assert diagnostics["skipped_no_price"] == 1
    assert diagnostics["skipped_one_lot_gt_target"] == 1
    assert diagnostics["skipped_min_notional"] == 0
    assert diagnostics["actual_notional_sum"] == 49_500.0
    assert math.isclose(diagnostics["abs_weight_error_sum"], 0.505)


def test_allocate_round_lot_respects_min_notional_and_cash_ordering():
    targets = pd.DataFrame(
        [
            {"symbol": "large", "target_weight": 0.80},
            {"symbol": "small", "target_weight": 0.04},
        ]
    )
    prices = pd.Series({"large": 10.0, "small": 10.0})

    allocation, diagnostics = allocate_round_lot(
        targets,
        prices,
        equity=100_000.0,
        round_lot=100,
        min_notional=5_000.0,
        max_weight=0.8,
    )

    assert allocation == {"large": 8_000}
    assert diagnostics["skipped_min_notional"] == 1
    assert diagnostics["actual_notional_sum"] == 80_000.0


def test_allocate_round_lot_account_skips_high_price_and_leaves_cap_cash():
    targets = pd.DataFrame(
        [
            {"symbol": "cheap", "target_weight": 0.50},
            {"symbol": "high", "target_weight": 0.50},
        ]
    )
    prices = pd.Series({"cheap": 10.0, "high": 1_000.0})

    allocation, diagnostics = allocate_round_lot_account(
        targets,
        prices,
        equity=100_000.0,
        round_lot=100,
        min_notional=5_000.0,
        max_weight=0.60,
    )

    assert allocation == {"cheap": 6_000}
    assert diagnostics["skipped_high_price"] == 1
    assert diagnostics["eligible_names"] == 1
    assert diagnostics["target_weight_sum"] == 0.60
    assert diagnostics["actual_notional_sum"] == 60_000.0
    assert diagnostics["cash_left"] == 40_000.0


def test_allocate_round_lot_account_redistributes_remaining_cash():
    targets = pd.DataFrame(
        [
            {"symbol": "a", "target_weight": 0.50},
            {"symbol": "b", "target_weight": 0.50},
        ]
    )
    prices = pd.Series({"a": 13.0, "b": 17.0})

    allocation, diagnostics = allocate_round_lot_account(
        targets,
        prices,
        equity=100_000.0,
        round_lot=100,
        min_notional=5_000.0,
        max_weight=0.60,
    )

    assert allocation == {"a": 3_900, "b": 2_900}
    assert diagnostics["redistribution_rounds"] == 1
    assert diagnostics["actual_notional_sum"] == 100_000.0
    assert diagnostics["cash_left"] == 0.0
    assert diagnostics["max_actual_weight"] <= 0.60


def test_portfolio_value_ignores_missing_prices():
    value = portfolio_value({"a": 100, "b": 200}, pd.Series({"a": 10.0}), cash=1_000.0)

    assert value == 2_000.0
