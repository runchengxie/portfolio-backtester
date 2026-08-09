"""Stage 2 fixed-scenario contract tests for the optional ledger switch.

These tests pin the behavior of the opt-in ``ledger`` mode on the ``native``
position-replay backend and ``backtest_topk``. The default contract must stay
unchanged (``not_available`` declarations, five-element ``backtest_topk``
bundle). The ledger switch wires the independent execution-sim engine into the
shared eight-field reconciled ledger.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_backtester.api import backtest_topk
from portfolio_backtester.backends.native import (
    NativePositionReplayBackend,
    NativePositionReplayRequest,
)
from portfolio_backtester.execution_sim import (
    ExecutionSimConfig,
    UnifiedLedger,
    simulate_execution_adjusted_nav,
    simulate_ideal_daily_nav,
)
from portfolio_backtester.position_backtest import PositionBacktestConfig


def _pricing_frame(dates, symbols, *, amount=500_000.0, price=10.0, tradable=True):
    rows = []
    for i, date in enumerate(pd.to_datetime(dates)):
        for symbol in symbols:
            rows.append(
                {
                    "trade_date": date,
                    "symbol": symbol,
                    "open": float(price) + i,
                    "amount": float(amount),
                    "medadv20_amount": float(amount),
                    "is_tradable": bool(tradable),
                }
            )
    return pd.DataFrame(rows)


def _positions(rebalance_date, entry_date, symbols, weight):
    return pd.DataFrame(
        {
            "rebalance_date": [rebalance_date] * len(symbols),
            "entry_date": [entry_date] * len(symbols),
            "symbol": symbols,
            "weight": [weight] * len(symbols),
            "side": ["long"] * len(symbols),
        }
    )


# --------------------------------------------------------------------------- #
# to_unified_ledger adapter
# --------------------------------------------------------------------------- #


def test_unified_ledger_has_eight_fields_and_conserves():
    dates = pd.date_range("2020-01-02", periods=4, freq="B")
    positions = _positions("20200102", "20200102", ["AAA"], 1.0)
    pricing = _pricing_frame(dates, ["AAA"])
    result = simulate_ideal_daily_nav(
        positions,
        pricing,
        price_col="open",
        transaction_cost_bps=0.0,
        portfolio_value=1_000.0,
    )
    ledger = result.to_unified_ledger(portfolio_value=1_000.0)

    assert isinstance(ledger, UnifiedLedger)
    for field in (
        "targets",
        "orders",
        "fills",
        "daily_positions",
        "daily_cash",
        "daily_nav",
        "cost_breakdown",
        "turnover_breakdown",
    ):
        assert getattr(ledger, field).shape[0] >= 0

    nav = ledger.daily_nav["nav"].to_numpy(dtype=float)
    cash = ledger.daily_cash["cash"].to_numpy(dtype=float)
    positions_value = ledger.daily_positions["positions_value"].to_numpy(dtype=float)
    assert np.allclose(nav, cash + positions_value, rtol=1e-9)
    assert ledger.daily_nav["trade_date"].equals(ledger.daily_cash["trade_date"])
    assert ledger.daily_cash["trade_date"].equals(ledger.daily_positions["trade_date"])


def test_unified_ledger_cost_turnover_breakdown_by_side():
    dates = pd.date_range("2020-01-02", periods=3, freq="B")
    positions = _positions("20200102", "20200102", ["AAA", "BBB"], 0.5)
    pricing = _pricing_frame(dates, ["AAA", "BBB"])
    result = simulate_ideal_daily_nav(
        positions,
        pricing,
        price_col="open",
        transaction_cost_bps=10.0,
        portfolio_value=10_000.0,
    )
    ledger = result.to_unified_ledger(portfolio_value=10_000.0)

    total_cost = float(
        ledger.cost_breakdown.loc[
            ledger.cost_breakdown["side"].eq("total"), "transaction_cost"
        ].iloc[0]
    )
    side_cost = ledger.cost_breakdown[ledger.cost_breakdown["side"].ne("total")][
        "transaction_cost"
    ].sum()
    assert total_cost == pytest.approx(side_cost, rel=1e-9)
    assert total_cost > 0.0

    total_turn = float(
        ledger.turnover_breakdown.loc[
            ledger.turnover_breakdown["side"].eq("total"), "filled_notional"
        ].iloc[0]
    )
    side_turn = ledger.turnover_breakdown[ledger.turnover_breakdown["side"].ne("total")][
        "filled_notional"
    ].sum()
    assert total_turn == pytest.approx(side_turn, rel=1e-9)


# --------------------------------------------------------------------------- #
# native backend optional ledger switch
# --------------------------------------------------------------------------- #


def _native_request(ledger: bool):
    dates = pd.date_range("2020-01-02", periods=4, freq="B")
    positions = _positions("20200102", "20200102", ["AAA", "BBB"], 0.5)
    pricing = _pricing_frame(dates, ["AAA", "BBB"])
    periods = pd.DataFrame(
        [
            {
                "rebalance_date": "20200102",
                "entry_date": "20200102",
                "exit_date": dates[-1].strftime("%Y%m%d"),
            }
        ]
    )
    config = PositionBacktestConfig(price_col="open", transaction_cost_bps=0.0)
    return NativePositionReplayRequest(
        positions=positions,
        pricing=pricing,
        periods=periods,
        config=config,
        ledger=ledger,
    )


def test_native_default_contract_unchanged():
    result = NativePositionReplayBackend().run(_native_request(ledger=False))
    assert result.capabilities.daily_ledger is False
    assert result.capabilities.order_lifecycle is False
    assert result.metadata["daily_ledger"] == "not_available"
    assert result.metadata["orders_and_fills"] == "not_available"
    assert result.orders.empty
    assert result.fills.empty
    assert result.daily_ledger.empty


def test_native_ledger_flips_capability_and_conserves():
    result = NativePositionReplayBackend().run(_native_request(ledger=True))
    assert result.capabilities.daily_ledger is True
    assert result.capabilities.order_lifecycle is True
    assert result.metadata["daily_ledger"] == "execution_sim"

    assert not result.orders.empty
    assert not result.fills.empty
    assert not result.daily_ledger.empty

    # CanonicalBacktestResult contract requires stable id columns.
    assert "order_id" in result.orders.columns
    assert "fill_id" in result.fills.columns
    assert "order_id" in result.fills.columns

    nav = result.daily_ledger["nav"].to_numpy(dtype=float)
    cash = result.daily_ledger["cash"].to_numpy(dtype=float)
    positions_value = result.daily_ledger["positions_value"].to_numpy(dtype=float)
    assert np.allclose(nav, cash + positions_value, rtol=1e-9)


def test_native_ledger_matches_adjusted_nav_directly():
    dates = pd.date_range("2020-01-02", periods=4, freq="B")
    positions = _positions("20200102", "20200102", ["AAA", "BBB"], 0.5)
    pricing = _pricing_frame(dates, ["AAA", "BBB"])
    result = simulate_execution_adjusted_nav(
        positions,
        pricing,
        ExecutionSimConfig(enabled=True, portfolio_value=1_000_000.0),
        price_col="open",
        transaction_cost_bps=0.0,
        trading_days_per_year=252,
    )
    direct = result.to_unified_ledger(portfolio_value=1_000_000.0)

    backend = NativePositionReplayBackend().run(_native_request(ledger=True))
    backend_nav = backend.daily_ledger["nav"].round(6).tolist()
    direct_nav = direct.daily_nav["nav"].round(6).tolist()
    assert backend_nav == direct_nav


# --------------------------------------------------------------------------- #
# backtest_topk optional ledger switch
# --------------------------------------------------------------------------- #


def _topk_data_and_pricing():
    dates = pd.date_range("2020-01-02", periods=6, freq="B")
    rebs = [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-06")]
    data = pd.DataFrame(
        [
            {
                "trade_date": d,
                "symbol": s,
                "pred": 3.0 if s == "AAA" else (2.0 if s == "BBB" else 1.0),
            }
            for i, d in enumerate(dates)
            for s in ["AAA", "BBB", "CCC"]
        ]
    )
    pricing = _pricing_frame(dates, ["AAA", "BBB", "CCC"])
    return data, pricing, rebs


def test_backtest_topk_default_bundle_unchanged():
    data, pricing, rebs = _topk_data_and_pricing()
    result = backtest_topk(
        data,
        "pred",
        "open",
        rebs,
        top_k=2,
        shift_days=0,
        cost_bps=0.0,
        trading_days_per_year=252,
        pricing_data=pricing,
    )
    assert len(result) == 5


def test_backtest_topk_ledger_appends_unified_ledger():
    data, pricing, rebs = _topk_data_and_pricing()
    result = backtest_topk(
        data,
        "pred",
        "open",
        rebs,
        top_k=2,
        shift_days=0,
        cost_bps=0.0,
        trading_days_per_year=252,
        pricing_data=pricing,
        ledger=True,
    )
    assert len(result) == 6
    _stats, _net, _gross, _turnover, _period_info, ledger = result
    assert isinstance(ledger, UnifiedLedger)

    nav = ledger.daily_nav["nav"].to_numpy(dtype=float)
    cash = ledger.daily_cash["cash"].to_numpy(dtype=float)
    positions_value = ledger.daily_positions["positions_value"].to_numpy(dtype=float)
    assert np.allclose(nav, cash + positions_value, rtol=1e-9)


def test_backtest_topk_ledger_equivalent_to_ideal_sim_of_targets():
    data, pricing, rebs = _topk_data_and_pricing()
    result = backtest_topk(
        data,
        "pred",
        "open",
        rebs,
        top_k=2,
        shift_days=0,
        cost_bps=0.0,
        trading_days_per_year=252,
        pricing_data=pricing,
        ledger=True,
    )
    ledger = result[5]

    # Rebuild the same targets the engine selected and run ideal_daily_nav directly.
    targets = ledger.targets.copy()
    positions = targets.rename(columns={"target_weight": "weight"})[
        ["rebalance_date", "entry_date", "symbol", "weight", "side"]
    ]
    direct = simulate_ideal_daily_nav(
        positions,
        pricing,
        price_col="open",
        transaction_cost_bps=0.0,
        portfolio_value=1_000_000.0,
    ).to_unified_ledger(portfolio_value=1_000_000.0)

    assert ledger.daily_nav["nav"].round(6).tolist() == direct.daily_nav["nav"].round(6).tolist()


# --------------------------------------------------------------------------- #
# order-level minimum commission
# --------------------------------------------------------------------------- #


def test_unified_ledger_respects_minimum_commission():
    from portfolio_backtester.execution_sim import TradeFeeModel

    dates = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])
    positions = _positions("20200102", "20200102", ["AAA", "BBB"], 0.01)
    pricing = _pricing_frame(dates, ["AAA", "BBB"], amount=500_000.0, price=10.0)
    fee_model = TradeFeeModel(
        buy_commission_bps=1.0,
        sell_commission_bps=1.0,
        min_commission=5.0,
    )
    result = simulate_ideal_daily_nav(
        positions,
        pricing,
        price_col="open",
        transaction_cost_bps=0.0,
        portfolio_value=10_000.0,
        trade_fee_model=fee_model,
    )
    ledger = result.to_unified_ledger(portfolio_value=10_000.0)

    assert not ledger.fills.empty
    assert float(ledger.fills["transaction_cost"].min()) >= 5.0 - 1e-9
    total_cost = float(
        ledger.cost_breakdown.loc[
            ledger.cost_breakdown["side"].eq("total"), "transaction_cost"
        ].iloc[0]
    )
    assert total_cost >= 2.0 * 5.0 - 1e-9
