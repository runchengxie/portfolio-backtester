"""Phase 4 market-rule contract tests for the execution simulation engine.

These tests exercise the opt-in A-share market rules (整手买入/零股卖出, T+1,
涨跌停, 上市/停牌/退市) and the audit timestamps added in phase 4. They use
small synthetic panels and never touch real data. All market rules default to
OFF, so the locked fixed-comparison scenarios are unaffected.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from portfolio_backtester.execution_sim import (
    ExecutionSimConfig,
    simulate_execution_adjusted_nav,
)


def _pricing_frame(dates, symbols, *, amount=500_000.0, price=10.0):
    rows = []
    for date in pd.to_datetime(dates):
        for symbol in symbols:
            rows.append(
                {
                    "trade_date": date,
                    "symbol": symbol,
                    "open": float(price),
                    "amount": float(amount),
                    "medadv20_amount": float(amount),
                }
            )
    return pd.DataFrame(rows)


def _single_buy_position(rebalance="20200101", entry="20200102", symbol="AAA", weight=0.10):
    return pd.DataFrame(
        {
            "rebalance_date": [rebalance],
            "entry_date": [entry],
            "symbol": [symbol],
            "weight": [weight],
            "side": ["long"],
        }
    )


def _config(**overrides):
    base = {
        "enabled": True,
        "portfolio_value": 1_000_000.0,
        "participation_rate": 0.50,
        "liquidity_cols": ("amount",),
        "buy_max_days": 5,
    }
    base.update(overrides)
    return ExecutionSimConfig(**base)


def test_warnings_recorded_when_market_rules_inactive():
    dates = pd.date_range("2020-01-01", periods=3, freq="B")
    config = _config()  # 全部市场规则默认关闭
    result = simulate_execution_adjusted_nav(
        _single_buy_position(),
        _pricing_frame(dates, ["AAA"]),
        config,
        price_col="open",
    )
    assert result.summary["warnings"] == [
        "market_rules_inactive: A-share lot/T+1/limit/listing rules not enforced"
    ]


def test_no_warning_when_round_lot_active():
    dates = pd.date_range("2020-01-01", periods=3, freq="B")
    config = _config(round_lot=100)
    result = simulate_execution_adjusted_nav(
        _single_buy_position(),
        _pricing_frame(dates, ["AAA"]),
        config,
        price_col="open",
    )
    assert result.summary["warnings"] == []


def test_round_lot_buy_quantizes_to_whole_lots():
    # 100万组合, 10% 权重 = 10万 notional, 价格 10 -> 10000 股, 整手后应为 10000 (整除).
    dates = pd.date_range("2020-01-01", periods=3, freq="B")
    config = _config(round_lot=100)
    result = simulate_execution_adjusted_nav(
        _single_buy_position(weight=0.10),
        _pricing_frame(dates, ["AAA"], price=10.0),
        config,
        price_col="open",
    )
    fills = result.fills
    assert not fills.empty
    buy_fills = fills[fills["side"] == "buy"]
    assert (buy_fills["filled_notional"] > 0).any()
    # 成交数量 = notional / price 必须是 100 的整数倍.
    for _, row in buy_fills.iterrows():
        # 由 fill 反推数量: shares 增量未在 fills 单列, 这里用 notional 与 price 校验取整.
        # 价格恒为 10, 故 filled_notional / 10 应为整百.
        qty = row["filled_notional"] / 10.0
        assert abs(qty - round(qty / 100.0) * 100.0) < 1e-6


def test_round_lot_buy_skips_sub_lot():
    # 权重 0.001 -> 1000 notional -> 100 股恰好一手, 仍然成交; 用 0.0005 -> 50 股不足一手, 不买.
    dates = pd.date_range("2020-01-01", periods=3, freq="B")
    config = _config(round_lot=100)
    result = simulate_execution_adjusted_nav(
        _single_buy_position(weight=0.0005),
        _pricing_frame(dates, ["AAA"], price=10.0),
        config,
        price_col="open",
    )
    buy_fills = result.fills[result.fills["side"] == "buy"]
    # 不足一手 -> 当日不成交.
    assert buy_fills.empty or (buy_fills["filled_notional"] <= 1e-6).all()


def test_odd_lot_sell_allowed():
    # 卖出路径不受整手约束: 先整手买入 10000 股, 再减仓到 9500 股 (卖出 500 零股),
    # 卖出应全部成交 (卖出不限整手).
    dates = pd.date_range("2020-01-01", periods=5, freq="B")
    positions = pd.DataFrame(
        {
            "rebalance_date": ["20200101", "20200103"],
            "entry_date": ["20200102", "20200106"],
            "symbol": ["AAA", "AAA"],
            "weight": [0.10, 0.095],
            "side": ["long", "long"],
        }
    )
    config = _config(round_lot=100, buy_max_days=1, sell_max_days=1)
    pricing = _pricing_frame(dates, ["AAA"], price=10.0)
    result = simulate_execution_adjusted_nav(
        positions,
        pricing,
        config,
        price_col="open",
    )
    sell_fills = result.fills[result.fills["side"] == "sell"]
    # 卖出应发生 (零股卖出路径不强制整手), 且卖出数量对应 500 股 (零股).
    assert not sell_fills.empty
    total_sold = float(sell_fills["filled_notional"].sum()) / 10.0
    assert total_sold > 0
    # 500 股零股被允许卖出 (未被整手约束拦截).
    assert math.isclose(total_sold, 500.0, rel_tol=1e-6)


def test_t1_allows_next_day_sell_of_prior_holdings():
    # T+1: 当日买入的份额次日才可卖. 这里 day1 买入, day2 减仓 -> day2 卖出应成交
    # (持仓自 day1 起, 满足 T+1). 验证 T+1 不会误伤合法的次日卖出.
    dates = pd.date_range("2020-01-01", periods=4, freq="B")
    positions = pd.DataFrame(
        {
            "rebalance_date": ["20200101", "20200103"],
            "entry_date": ["20200102", "20200106"],
            "symbol": ["AAA", "AAA"],
            "weight": [0.10, 0.0],
            "side": ["long", "long"],
        }
    )
    config = _config(enforce_t1=True, round_lot=100, buy_max_days=1, sell_max_days=1)
    pricing = _pricing_frame(dates, ["AAA"], price=10.0)
    result = simulate_execution_adjusted_nav(
        positions,
        pricing,
        config,
        price_col="open",
    )
    sell_fills = result.fills[result.fills["side"] == "sell"]
    # day2 卖出 day1 买入的份额 -> 应全部成交.
    assert not sell_fills.empty
    assert (sell_fills["filled_notional"] > 0).any()
    total_sold = float(sell_fills["filled_notional"].sum()) / 10.0
    assert math.isclose(total_sold, 10_000.0, rel_tol=1e-6)


def test_limit_up_blocks_buy():
    dates = pd.date_range("2020-01-01", periods=3, freq="B")
    pricing = _pricing_frame(dates, ["AAA"])
    pricing["limit_up"] = False
    pricing.loc[pricing["trade_date"] == dates[0], "limit_up"] = True
    config = _config(enforce_price_limits=True, round_lot=100, buy_max_days=1)
    result = simulate_execution_adjusted_nav(
        _single_buy_position(),
        pricing,
        config,
        price_col="open",
        limit_up_col="limit_up",
        limit_down_col="limit_up",  # 复用同一列; limit_down 在买入路径不使用
    )
    day0 = dates[0].strftime("%Y%m%d")
    day0_buys = result.fills[
        (result.fills["side"] == "buy") & (result.fills["trade_date"] == day0)
    ]
    assert day0_buys.empty or (day0_buys["filled_notional"] <= 1e-6).all()
    # 涨停次日可买.
    later_buys = result.fills[
        (result.fills["side"] == "buy") & (result.fills["trade_date"] != day0)
    ]
    assert (later_buys["filled_notional"] > 0).any()


def test_limit_down_blocks_sell():
    dates = pd.date_range("2020-01-01", periods=5, freq="B")
    positions = pd.DataFrame(
        {
            "rebalance_date": ["20200101", "20200103"],
            "entry_date": ["20200102", "20200106"],
            "symbol": ["AAA", "AAA"],
            "weight": [0.10, -0.10],
            "side": ["long", "long"],
        }
    )
    pricing = _pricing_frame(dates, ["AAA"])
    pricing["limit_down"] = False
    pricing.loc[pricing["trade_date"] == dates[1], "limit_down"] = True
    config = _config(enforce_price_limits=True, round_lot=100, buy_max_days=1, sell_max_days=2)
    result = simulate_execution_adjusted_nav(
        positions,
        pricing,
        config,
        price_col="open",
        limit_up_col="limit_down",
        limit_down_col="limit_down",
    )
    day1_sells = result.fills[
        (result.fills["side"] == "sell")
        & (result.fills["trade_date"] == dates[1].strftime("%Y%m%d"))
    ]
    assert day1_sells.empty or (day1_sells["filled_notional"] <= 1e-6).all()


def test_listing_status_halt_skips_trading():
    dates = pd.date_range("2020-01-01", periods=3, freq="B")
    pricing = _pricing_frame(dates, ["AAA"])
    pricing["listing_status"] = "listed"
    pricing.loc[pricing["trade_date"] == dates[0], "listing_status"] = "halted"
    config = _config(enforce_listing_status=True, round_lot=100, buy_max_days=1)
    result = simulate_execution_adjusted_nav(
        _single_buy_position(),
        pricing,
        config,
        price_col="open",
        listing_status_col="listing_status",
    )
    day0_fills = result.fills[result.fills["trade_date"] == dates[0].strftime("%Y%m%d")]
    assert day0_fills.empty or (day0_fills["filled_notional"] <= 1e-6).all()
    later_fills = result.fills[result.fills["trade_date"] != dates[0].strftime("%Y%m%d")]
    assert (later_fills["filled_notional"] > 0).any()


def test_enforce_price_limits_without_column_raises():
    dates = pd.date_range("2020-01-01", periods=3, freq="B")
    config = _config(enforce_price_limits=True, round_lot=100)
    with pytest.raises(ValueError, match="enforce_price_limits"):
        simulate_execution_adjusted_nav(
            _single_buy_position(),
            _pricing_frame(dates, ["AAA"]),
            config,
            price_col="open",
        )


def test_enforce_listing_status_without_column_raises():
    dates = pd.date_range("2020-01-01", periods=3, freq="B")
    config = _config(enforce_listing_status=True, round_lot=100)
    with pytest.raises(ValueError, match="enforce_listing_status"):
        simulate_execution_adjusted_nav(
            _single_buy_position(),
            _pricing_frame(dates, ["AAA"]),
            config,
            price_col="open",
        )


def test_audit_timestamps_present_with_timezone():
    dates = pd.date_range("2020-01-01", periods=3, freq="B")
    config = _config(round_lot=100)
    result = simulate_execution_adjusted_nav(
        _single_buy_position(),
        _pricing_frame(dates, ["AAA"]),
        config,
        price_col="open",
    )
    fills = result.fills
    assert not fills.empty
    required = {"signal_time", "decision_time", "order_time", "fill_time", "valuation_time"}
    assert required.issubset(fills.columns)
    for col in required:
        vals = fills[col].dropna()
        assert not vals.empty, f"{col} 应有审计时间戳"
        # 时区应为 +08:00 (Asia/Shanghai).
        assert all("+08:00" in str(v) for v in vals), f"{col} 缺少 +08:00 时区"
    # 信号时间应取 rebalance_date (20200101).
    assert fills["signal_time"].dropna().str.startswith("2020-01-01").all()


def test_default_call_preserves_numeric_columns():
    # 默认 (无规则) 调用应仍产出与原先一致的结构, 不抛错, 且时间戳列为 None.
    dates = pd.date_range("2020-01-01", periods=3, freq="B")
    config = _config()
    result = simulate_execution_adjusted_nav(
        _single_buy_position(),
        _pricing_frame(dates, ["AAA"]),
        config,
        price_col="open",
    )
    assert result.summary["status"] == "ok"
    for col in {"signal_time", "decision_time", "order_time", "fill_time", "valuation_time"}:
        assert col in result.fills.columns
