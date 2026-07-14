from __future__ import annotations

import pandas as pd
import pytest

from portfolio_backtester.types import BacktestLegResult, CostBreakdown


def test_cost_breakdown_reports_components_and_total() -> None:
    breakdown = CostBreakdown(fee_cost=0.001, slippage_cost=0.002)

    assert breakdown.total_cost == pytest.approx(0.003)
    assert breakdown.to_dict() == {
        "fee_cost": pytest.approx(0.001),
        "slippage_cost": pytest.approx(0.002),
        "total_cost": pytest.approx(0.003),
    }


def test_backtest_leg_result_exposes_net_cost_and_turnover_breakdown() -> None:
    result = BacktestLegResult(
        holdings=["A"],
        weights=pd.Series({"A": 1.0}),
        entry_prices=pd.Series({"A": 10.0}),
        exit_idx=1,
        exit_date=pd.Timestamp("2026-01-02"),
        gross=0.02,
        turnover=0.75,
        fee_cost=0.001,
        slippage_cost=0.002,
        buy_turnover=0.75,
        sell_turnover=0.75,
        gross_traded_weight=1.5,
        half_l1_turnover=0.75,
    )

    assert result.turnover_breakdown.gross_traded_weight == pytest.approx(1.5)
    assert result.total_cost == pytest.approx(0.003)
    assert result.net == pytest.approx(0.017)
    assert result.cost_breakdown.to_dict()["total_cost"] == pytest.approx(0.003)
