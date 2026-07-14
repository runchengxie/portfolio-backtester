from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_backtester.execution_sim import (
    ExecutionSimConfig,
    TradeFeeModel,
    build_execution_sim_config,
    required_execution_sim_columns,
    simulate_capacity_execution,
    simulate_execution_adjusted_nav,
    simulate_ideal_daily_nav,
)


def _pricing_frame(dates, symbols, *, amount_map=None, tradable_map=None, price_map=None):
    rows = []
    amount_map = amount_map or {}
    tradable_map = tradable_map or {}
    price_map = price_map or {}
    for date in pd.to_datetime(dates):
        for symbol in symbols:
            amount = float(amount_map.get((symbol, date.strftime("%Y%m%d")), 500_000.0))
            price = float(price_map.get((symbol, date.strftime("%Y%m%d")), 10.0))
            rows.append(
                {
                    "trade_date": date,
                    "symbol": symbol,
                    "open": price,
                    "amount": amount,
                    "medadv20_amount": amount,
                    "is_tradable": bool(
                        tradable_map.get((symbol, date.strftime("%Y%m%d")), amount > 0)
                    ),
                }
            )
    return pd.DataFrame(rows)


def test_capacity_execution_uses_side_aware_buy_tradeability():
    dates = pd.date_range("2020-01-01", periods=3, freq="B")
    positions = pd.DataFrame(
        {
            "rebalance_date": ["20200101"],
            "entry_date": ["20200102"],
            "symbol": ["AAA"],
            "weight": [0.10],
            "side": ["long"],
        }
    )
    pricing = _pricing_frame(dates, ["AAA"])
    pricing["is_buy_tradable"] = False
    pricing["is_sell_tradable"] = True
    config = ExecutionSimConfig(
        enabled=True,
        portfolio_value=1_000_000.0,
        participation_rate=0.10,
        liquidity_cols=("amount",),
        buy_max_days=1,
    )

    result = simulate_capacity_execution(
        positions,
        pricing,
        config,
        price_col="open",
        tradable_col="is_tradable",
        buy_tradable_col="is_buy_tradable",
        sell_tradable_col="is_sell_tradable",
    )

    assert result.orders.loc[0, "side"] == "buy"
    assert result.orders.loc[0, "filled_weight"] == 0.0
    assert result.orders.loc[0, "status"] == "cancelled_buy_deadline"


def test_capacity_execution_uses_side_aware_sell_tradeability():
    dates = pd.date_range("2020-01-01", periods=5, freq="B")
    positions = pd.DataFrame(
        {
            "rebalance_date": ["20200101", "20200103"],
            "entry_date": ["20200102", "20200106"],
            "symbol": ["AAA", "BBB"],
            "weight": [0.10, 0.10],
            "side": ["long", "long"],
        }
    )
    pricing = _pricing_frame(dates, ["AAA", "BBB"])
    pricing["is_buy_tradable"] = True
    pricing["is_sell_tradable"] = pricing["symbol"].ne("AAA")
    config = ExecutionSimConfig(
        enabled=True,
        portfolio_value=1_000_000.0,
        participation_rate=0.10,
        liquidity_cols=("amount",),
        buy_max_days=1,
        sell_max_days=1,
    )

    result = simulate_capacity_execution(
        positions,
        pricing,
        config,
        price_col="open",
        tradable_col="is_tradable",
        buy_tradable_col="is_buy_tradable",
        sell_tradable_col="is_sell_tradable",
    )

    sell_order = result.orders[result.orders["side"] == "sell"].iloc[0]
    assert sell_order["symbol"] == "AAA"
    assert sell_order["filled_weight"] == 0.0
    assert sell_order["status"] == "delayed_sell"


def test_capacity_execution_partially_fills_buy_deadline():
    dates = pd.date_range("2020-01-01", periods=7, freq="B")
    positions = pd.DataFrame(
        {
            "rebalance_date": ["20200101", "20200101"],
            "entry_date": ["20200102", "20200102"],
            "symbol": ["AAA", "BBB"],
            "weight": [0.10, 0.10],
            "side": ["long", "long"],
        }
    )
    amount_map = {}
    for date in dates:
        amount_map[("AAA", date.strftime("%Y%m%d"))] = 500_000.0
        amount_map[("BBB", date.strftime("%Y%m%d"))] = 100_000.0
    pricing = _pricing_frame(dates, ["AAA", "BBB"], amount_map=amount_map)
    config = ExecutionSimConfig(
        enabled=True,
        portfolio_value=1_000_000.0,
        participation_rate=0.05,
        liquidity_cols=("medadv20_amount", "amount"),
        buy_max_days=5,
    )

    result = simulate_capacity_execution(
        positions,
        pricing,
        config,
        price_col="open",
        tradable_col="is_tradable",
    )

    orders = result.orders.set_index("symbol")
    assert orders.loc["AAA", "status"] == "filled"
    assert orders.loc["AAA", "filled_weight"] == pytest.approx(0.10)
    assert orders.loc["BBB", "status"] == "cancelled_buy_deadline"
    assert orders.loc["BBB", "filled_weight"] == pytest.approx(0.025)
    assert result.summary["unfilled_buy_notional"] == pytest.approx(75_000.0)


def test_capacity_execution_abandons_zero_fill_buy_after_threshold():
    dates = pd.date_range("2020-01-01", periods=7, freq="B")
    positions = pd.DataFrame(
        {
            "rebalance_date": ["20200101"],
            "entry_date": ["20200102"],
            "symbol": ["AAA"],
            "weight": [0.10],
            "side": ["long"],
        }
    )
    amount_map = {("AAA", date.strftime("%Y%m%d")): 0.0 for date in dates}
    pricing = _pricing_frame(dates, ["AAA"], amount_map=amount_map)
    config = ExecutionSimConfig(
        enabled=True,
        portfolio_value=1_000_000.0,
        participation_rate=0.05,
        liquidity_cols=("medadv20_amount", "amount"),
        buy_max_days=5,
        zero_fill_abort_days_buy=3,
    )

    result = simulate_capacity_execution(
        positions,
        pricing,
        config,
        price_col="open",
        tradable_col="is_tradable",
    )

    assert result.orders.loc[0, "status"] == "abandoned_zero_fill"
    assert result.orders.loc[0, "zero_fill_days"] == 3
    assert result.orders.loc[0, "filled_weight"] == 0.0
    assert result.summary["abandoned_buy_orders"] == 1
    assert result.fills.empty


def test_capacity_execution_keeps_unfilled_sell_as_delayed_exit():
    dates = pd.date_range("2020-01-01", periods=9, freq="B")
    positions = pd.DataFrame(
        {
            "rebalance_date": ["20200101", "20200106"],
            "entry_date": ["20200102", "20200107"],
            "symbol": ["AAA", "BBB"],
            "weight": [0.20, 0.20],
            "side": ["long", "long"],
        }
    )
    amount_map = {}
    for date in dates:
        amount_map[("AAA", date.strftime("%Y%m%d"))] = 2_000_000.0
        amount_map[("BBB", date.strftime("%Y%m%d"))] = 2_000_000.0
    for date in dates[4:]:
        amount_map[("AAA", date.strftime("%Y%m%d"))] = 0.0
    pricing = _pricing_frame(dates, ["AAA", "BBB"], amount_map=amount_map)
    config = ExecutionSimConfig(
        enabled=True,
        portfolio_value=1_000_000.0,
        participation_rate=0.10,
        liquidity_cols=("medadv20_amount", "amount"),
        buy_max_days=2,
        sell_max_days=2,
    )

    result = simulate_capacity_execution(
        positions,
        pricing,
        config,
        price_col="open",
        tradable_col="is_tradable",
    )

    sell_orders = result.orders[result.orders["side"] == "sell"]
    assert sell_orders.shape[0] == 1
    assert sell_orders.iloc[0]["symbol"] == "AAA"
    assert sell_orders.iloc[0]["status"] == "delayed_sell"
    assert sell_orders.iloc[0]["unfilled_weight"] == pytest.approx(0.20)
    assert result.summary["delayed_sell_orders"] == 1


def test_build_execution_sim_config_defaults_to_daily_amount_cap():
    config = build_execution_sim_config(
        {"enabled": True, "liquidity_col": "medadv60_amount"},
        default_portfolio_value=2_000_000.0,
        default_liquidity_col="adv20_amount",
    )

    assert config.portfolio_value == pytest.approx(2_000_000.0)
    assert config.liquidity_cols == ("medadv60_amount", "amount")
    assert required_execution_sim_columns(
        config,
        price_col="open",
        tradable_col="is_tradable",
    ) == {"open", "medadv60_amount", "amount"}


def test_capacity_execution_skips_long_short_targets():
    positions = pd.DataFrame(
        {
            "rebalance_date": ["20200101"],
            "entry_date": ["20200102"],
            "symbol": ["AAA"],
            "weight": [-0.10],
            "side": ["short"],
        }
    )
    pricing = _pricing_frame(pd.date_range("2020-01-01", periods=3, freq="B"), ["AAA"])

    result = simulate_capacity_execution(
        positions,
        pricing,
        ExecutionSimConfig(enabled=True),
        price_col="open",
    )

    assert result.summary["status"] == "skipped_long_short_not_supported"
    assert np.isnan(result.summary["fill_ratio"])


def test_execution_adjusted_nav_tracks_fully_filled_position_return():
    dates = pd.date_range("2020-01-02", periods=2, freq="B")
    positions = pd.DataFrame(
        {
            "rebalance_date": ["20200102"],
            "entry_date": ["20200102"],
            "symbol": ["AAA"],
            "weight": [1.0],
            "side": ["long"],
        }
    )
    pricing = _pricing_frame(
        dates,
        ["AAA"],
        amount_map={("AAA", "20200102"): 10_000.0, ("AAA", "20200103"): 10_000.0},
        price_map={("AAA", "20200102"): 10.0, ("AAA", "20200103"): 11.0},
    )
    config = ExecutionSimConfig(
        enabled=True,
        portfolio_value=1_000.0,
        participation_rate=1.0,
        liquidity_cols=("amount",),
        buy_max_days=1,
    )

    result = simulate_execution_adjusted_nav(
        positions,
        pricing,
        config,
        price_col="open",
        tradable_col="is_tradable",
        transaction_cost_bps=0.0,
        trading_days_per_year=252,
    )

    assert result.summary["status"] == "ok"
    assert result.daily["executed_nav"].iloc[-1] == pytest.approx(1.10)
    assert result.daily["executed_return"].iloc[-1] == pytest.approx(0.10)
    assert result.summary["stats"]["total_return"] == pytest.approx(0.10)


def test_ideal_daily_nav_tracks_fully_invested_position_return():
    dates = pd.date_range("2020-01-02", periods=2, freq="B")
    positions = pd.DataFrame(
        {
            "rebalance_date": ["20200102"],
            "entry_date": ["20200102"],
            "symbol": ["AAA"],
            "weight": [1.0],
            "side": ["long"],
        }
    )
    pricing = _pricing_frame(
        dates,
        ["AAA"],
        price_map={("AAA", "20200102"): 10.0, ("AAA", "20200103"): 11.0},
    )

    result = simulate_ideal_daily_nav(
        positions,
        pricing,
        price_col="open",
        transaction_cost_bps=0.0,
        trading_days_per_year=252,
        portfolio_value=1_000.0,
    )

    assert result.summary["status"] == "ok"
    assert result.summary["mode"] == "ideal_daily_nav"
    assert result.daily["executed_nav"].iloc[-1] == pytest.approx(1.10)
    assert result.daily["executed_return"].iloc[-1] == pytest.approx(0.10)
    assert result.orders["status"].tolist() == ["filled"]
    assert result.summary["fill_ratio"] == pytest.approx(1.0)


def test_ideal_daily_nav_reserves_cash_for_transaction_costs():
    dates = pd.date_range("2020-01-02", periods=2, freq="B")
    positions = pd.DataFrame(
        {
            "rebalance_date": ["20200102"],
            "entry_date": ["20200102"],
            "symbol": ["AAA"],
            "weight": [1.0],
            "side": ["long"],
        }
    )
    pricing = _pricing_frame(
        dates,
        ["AAA"],
        price_map={("AAA", "20200102"): 10.0, ("AAA", "20200103"): 11.0},
    )

    result = simulate_ideal_daily_nav(
        positions,
        pricing,
        price_col="open",
        transaction_cost_bps=10.0,
        trading_days_per_year=252,
        portfolio_value=1_000.0,
    )

    assert result.summary["status"] == "ok"
    assert result.orders["status"].tolist() == ["filled"]
    assert result.summary["fill_ratio"] == pytest.approx(1.0)
    assert result.fills["filled_notional"].tolist() == pytest.approx([999.000999000999])
    assert result.fills["transaction_cost"].tolist() == pytest.approx([0.9990009990009991])
    assert result.daily["executed_nav"].iloc[0] == pytest.approx(0.999000999000999)
    assert result.daily["executed_nav"].iloc[-1] == pytest.approx(1.098901098901099)


def test_ideal_daily_nav_uses_detailed_side_fee_model():
    dates = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])
    positions = pd.DataFrame(
        {
            "rebalance_date": ["20200102", "20200103"],
            "entry_date": ["20200102", "20200103"],
            "symbol": ["AAA", "BBB"],
            "weight": [1.0, 1.0],
            "side": ["long", "long"],
        }
    )
    pricing = _pricing_frame(
        dates,
        ["AAA", "BBB"],
        price_map={
            ("AAA", "20200102"): 10.0,
            ("AAA", "20200103"): 10.0,
            ("AAA", "20200106"): 10.0,
            ("BBB", "20200102"): 20.0,
            ("BBB", "20200103"): 20.0,
            ("BBB", "20200106"): 20.0,
        },
    )
    fee_model = TradeFeeModel(
        buy_commission_bps=1.0,
        sell_commission_bps=1.0,
        sell_stamp_duty_bps=5.0,
        transfer_fee_bps=0.1,
        min_commission=5.0,
        buy_slippage_bps=2.0,
        sell_slippage_bps=3.0,
    )

    result = simulate_ideal_daily_nav(
        positions,
        pricing,
        price_col="open",
        transaction_cost_bps=0.0,
        trading_days_per_year=252,
        portfolio_value=10_000.0,
        trade_fee_model=fee_model,
    )

    assert result.summary["fee_model"] == {
        "name": "detailed",
        "buy_commission_bps": 1.0,
        "sell_commission_bps": 1.0,
        "sell_stamp_duty_bps": 5.0,
        "transfer_fee_bps": 0.1,
        "min_commission": 5.0,
        "buy_slippage_bps": 2.0,
        "sell_slippage_bps": 3.0,
        "portfolio_value": 10_000.0,
    }
    assert result.fills["transaction_cost"].tolist() == pytest.approx(
        [7.09851005785892, 13.094253080312976, 7.0942718428940275]
    )
    assert result.daily["transaction_cost"].tolist() == pytest.approx(
        [7.09851005785892, 20.188524923207003, 0.0]
    )


def test_ideal_daily_nav_rebalances_at_entry_price_before_next_day_mark():
    dates = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])
    positions = pd.DataFrame(
        {
            "rebalance_date": ["20200102", "20200103"],
            "entry_date": ["20200102", "20200103"],
            "symbol": ["AAA", "BBB"],
            "weight": [1.0, 1.0],
            "side": ["long", "long"],
        }
    )
    pricing = _pricing_frame(
        dates,
        ["AAA", "BBB"],
        price_map={
            ("AAA", "20200102"): 10.0,
            ("AAA", "20200103"): 12.0,
            ("AAA", "20200106"): 12.0,
            ("BBB", "20200102"): 20.0,
            ("BBB", "20200103"): 20.0,
            ("BBB", "20200106"): 22.0,
        },
    )

    result = simulate_ideal_daily_nav(
        positions,
        pricing,
        price_col="open",
        transaction_cost_bps=0.0,
        trading_days_per_year=252,
        portfolio_value=1_000.0,
    )

    assert result.daily["executed_nav"].tolist() == pytest.approx([1.0, 1.2, 1.32])
    assert result.daily["executed_return"].tolist() == pytest.approx([0.0, 0.2, 0.1])
    assert result.orders["side"].tolist() == ["buy", "sell", "buy"]
    assert result.fills["filled_notional"].tolist() == pytest.approx([1000.0, 1200.0, 1200.0])


def test_execution_adjusted_nav_keeps_cash_for_unfilled_buy():
    dates = pd.date_range("2020-01-02", periods=2, freq="B")
    positions = pd.DataFrame(
        {
            "rebalance_date": ["20200102"],
            "entry_date": ["20200102"],
            "symbol": ["AAA"],
            "weight": [1.0],
            "side": ["long"],
        }
    )
    pricing = _pricing_frame(
        dates,
        ["AAA"],
        amount_map={("AAA", "20200102"): 500.0, ("AAA", "20200103"): 500.0},
        price_map={("AAA", "20200102"): 10.0, ("AAA", "20200103"): 11.0},
    )
    config = ExecutionSimConfig(
        enabled=True,
        portfolio_value=1_000.0,
        participation_rate=1.0,
        liquidity_cols=("amount",),
        buy_max_days=1,
    )

    result = simulate_execution_adjusted_nav(
        positions,
        pricing,
        config,
        price_col="open",
        tradable_col="is_tradable",
        transaction_cost_bps=0.0,
        trading_days_per_year=252,
    )

    assert result.daily["cash_weight"].iloc[0] == pytest.approx(0.5)
    assert result.daily["executed_nav"].iloc[-1] == pytest.approx(1.05)
    assert result.summary["fill_ratio"] == pytest.approx(0.5)


def test_execution_adjusted_nav_splits_target_and_shortfall_cash():
    dates = pd.date_range("2020-01-02", periods=2, freq="B")
    positions = pd.DataFrame(
        {
            "rebalance_date": ["20200102"],
            "entry_date": ["20200102"],
            "symbol": ["AAA"],
            "weight": [0.5],
            "side": ["long"],
        }
    )
    pricing = _pricing_frame(
        dates,
        ["AAA"],
        amount_map={("AAA", "20200102"): 250.0, ("AAA", "20200103"): 250.0},
        price_map={("AAA", "20200102"): 10.0, ("AAA", "20200103"): 10.0},
    )
    config = ExecutionSimConfig(
        enabled=True,
        portfolio_value=1_000.0,
        participation_rate=1.0,
        liquidity_cols=("amount",),
        buy_max_days=1,
    )

    result = simulate_execution_adjusted_nav(
        positions,
        pricing,
        config,
        price_col="open",
        tradable_col="is_tradable",
        transaction_cost_bps=0.0,
        trading_days_per_year=252,
    )

    first_day = result.daily.iloc[0]
    assert first_day["cash_weight"] == pytest.approx(0.75)
    assert first_day["target_cash_weight"] == pytest.approx(0.50)
    assert first_day["execution_shortfall_cash_weight"] == pytest.approx(0.25)
    assert result.summary["avg_target_cash_weight"] == pytest.approx(0.50)
    assert result.summary["avg_execution_shortfall_cash_weight"] == pytest.approx(0.25)
