from __future__ import annotations

import pandas as pd
import pytest

from portfolio_backtester.a_share_round_lot import RoundLotVariant
from portfolio_backtester.a_share_round_lot_diagnostics import simulate_round_lot_variant


def test_simulate_round_lot_variant_preserves_probe_outputs() -> None:
    scored = pd.DataFrame(
        [
            {
                "trade_date": "20260601",
                "symbol": "000001.SZ",
                "signal_backtest": 0.9,
                "medadv20_amount": 100.0,
            },
            {
                "trade_date": "20260601",
                "symbol": "000002.SZ",
                "signal_backtest": 0.8,
                "medadv20_amount": 80.0,
            },
        ]
    )
    pricing = pd.DataFrame(
        [
            {"trade_date": "20260601", "symbol": "000001.SZ", "tr_close": 10.0},
            {"trade_date": "20260601", "symbol": "000002.SZ", "tr_close": 20.0},
            {"trade_date": "20260602", "symbol": "000001.SZ", "tr_close": 10.0},
            {"trade_date": "20260602", "symbol": "000002.SZ", "tr_close": 20.0},
            {"trade_date": "20260603", "symbol": "000001.SZ", "tr_close": 11.0},
            {"trade_date": "20260603", "symbol": "000002.SZ", "tr_close": 22.0},
        ]
    )
    industry = pd.DataFrame(
        [
            {"symbol": "000001.SZ", "first_industry_name": "Bank"},
            {"symbol": "000002.SZ", "first_industry_name": "Tech"},
        ]
    )
    variant = RoundLotVariant(
        target_holdings=2,
        liquidity_floor_q=0.0,
        weighting="equal",
        industry_cap=2,
        max_weight=0.6,
        min_notional=0.0,
    )

    summary, daily, diag = simulate_round_lot_variant(
        scored,
        pricing,
        industry,
        variant,
        capital=10_000.0,
        round_lot=100,
        cost_bps=10.0,
        oos_periods=None,
    )

    assert daily["trade_date"].tolist() == ["20260602", "20260603"]
    assert summary["variant"] == variant.name
    assert summary["trade_count"] == 2
    assert summary["rebalance_count"] == 1
    assert summary["avg_actual_holdings"] == 2.0
    assert summary["avg_trade_notional"] == 9_000.0
    assert daily.loc[0, "portfolio_value"] == pytest.approx(9_991.0)
    assert daily.loc[1, "portfolio_value"] == pytest.approx(10_891.0)
    assert diag.loc[0, "transaction_cost"] == pytest.approx(9.0)
    assert diag.loc[0, "cash_weight_after_trade"] == pytest.approx(991.0 / 9_991.0)
