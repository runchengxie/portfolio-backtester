from __future__ import annotations

from typing import cast

import pandas as pd
import pytest

from portfolio_backtester.turnover import build_rebalance_turnover_report
from portfolio_backtester.types import BacktestLegResult, CostBreakdown


def test_cost_breakdown_reports_components_and_total() -> None:
    breakdown = CostBreakdown(fee_cost=0.001, slippage_cost=0.002)

    assert breakdown.total_cost == pytest.approx(0.003)
    assert breakdown.to_dict() == {
        "fee_cost": pytest.approx(0.001),
        "slippage_cost": pytest.approx(0.002),
        "commission": pytest.approx(0.0),
        "stamp_tax": pytest.approx(0.0),
        "transfer_fee": pytest.approx(0.0),
        "spread_cost": pytest.approx(0.0),
        "temporary_impact": pytest.approx(0.0),
        "permanent_impact": pytest.approx(0.0),
        "opportunity_cost": pytest.approx(0.0),
        "financing_cost": pytest.approx(0.0),
        "total_cost": pytest.approx(0.003),
    }


def test_cost_breakdown_from_components_aggregates_consistently() -> None:
    breakdown = CostBreakdown.from_components(
        commission=0.001,
        stamp_tax=0.0005,
        transfer_fee=0.0001,
        spread_cost=0.002,
        temporary_impact=0.0003,
        permanent_impact=0.0002,
        opportunity_cost=0.0001,
        financing_cost=0.0004,
    )

    assert breakdown.fee_cost == pytest.approx(0.0016)
    assert breakdown.slippage_cost == pytest.approx(0.003)
    assert breakdown.total_cost == pytest.approx(0.0046)
    assert breakdown.total_cost == pytest.approx(
        breakdown.commission
        + breakdown.stamp_tax
        + breakdown.transfer_fee
        + breakdown.spread_cost
        + breakdown.temporary_impact
        + breakdown.permanent_impact
        + breakdown.opportunity_cost
        + breakdown.financing_cost
    )
    assert breakdown.to_dict()["total_cost"] == pytest.approx(0.0046)


def test_backtest_leg_result_exposes_net_cost_and_turnover_breakdown() -> None:
    result = BacktestLegResult(
        holdings=["A"],
        weights=pd.Series({"A": 1.0}),
        entry_prices=pd.Series({"A": 10.0}),
        exit_idx=1,
        exit_date=cast(pd.Timestamp, pd.Timestamp("2026-01-02")),
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


def test_rebalance_turnover_report_separates_target_demand_and_execution() -> None:
    previous = pd.Series(0.1, index=[f"S{i}" for i in range(10)])
    target = pd.Series(0.1, index=[*[f"S{i}" for i in range(8)], "N0", "N1"])
    symbols = previous.index.union(target.index)
    demand = target.reindex(symbols).fillna(0.0) - previous.reindex(symbols).fillna(0.0)

    report = build_rebalance_turnover_report(
        previous_holdings=previous.index,
        target_holdings=target.index,
        previous_target_weights=previous,
        target_weights=target,
        pretrade_trade_weights=demand,
    )

    assert report.target_name_turnover == pytest.approx(0.2)
    assert report.target_entered_names == ("N0", "N1")
    assert report.target_exited_names == ("S8", "S9")
    assert report.target_overlap_names == tuple(f"S{i}" for i in range(8))
    assert report.target_entered_count == 2
    assert report.target_exited_count == 2
    assert report.target_overlap_count == 8
    assert report.target_weight_full_l1 == pytest.approx(0.4)
    assert report.target_weight_half_l1 == pytest.approx(0.2)
    assert report.pretrade_demand_buy == pytest.approx(0.2)
    assert report.pretrade_demand_sell == pytest.approx(0.2)
    assert report.pretrade_demand_full_l1 == pytest.approx(0.4)
    assert report.pretrade_demand_half_l1 == pytest.approx(0.2)
    assert report.execution_data_available is False
    assert report.executed_buy is None
    assert report.executed_sell is None
    assert report.executed_gross is None
    assert report.executed_full_l1 is None
    assert report.executed_half_l1 is None
    assert report.executed_cost is None


def test_rebalance_turnover_report_marks_initial_build_without_changing_l1_units() -> None:
    target = pd.Series({"A": 0.5, "B": 0.5})

    report = build_rebalance_turnover_report(
        previous_holdings=None,
        target_holdings=target.index,
        previous_target_weights=None,
        target_weights=target,
        pretrade_trade_weights=target,
    )

    assert report.is_initial_build is True
    assert report.target_name_turnover == pytest.approx(1.0)
    assert report.target_entered_names == ("A", "B")
    assert report.target_exited_names == ()
    assert report.target_overlap_names == ()
    assert report.target_weight_full_l1 == pytest.approx(1.0)
    assert report.target_weight_half_l1 == pytest.approx(0.5)
    assert report.pretrade_demand_buy == pytest.approx(1.0)
    assert report.pretrade_demand_sell == pytest.approx(0.0)
    assert report.pretrade_demand_full_l1 == pytest.approx(1.0)
    assert report.pretrade_demand_half_l1 == pytest.approx(0.5)


def test_rebalance_turnover_report_reconciles_observed_execution() -> None:
    executed = pd.Series({"OLD": -0.1, "NEW": 0.1})
    report = build_rebalance_turnover_report(
        previous_holdings=["OLD"],
        target_holdings=["NEW"],
        previous_target_weights=pd.Series({"OLD": 1.0}),
        target_weights=pd.Series({"NEW": 1.0}),
        pretrade_trade_weights=pd.Series({"OLD": -1.0, "NEW": 1.0}),
        executed_trade_weights=executed,
        executed_cost=0.002,
    )

    assert report.execution_data_available is True
    assert report.executed_buy == pytest.approx(0.1)
    assert report.executed_sell == pytest.approx(0.1)
    assert report.executed_gross == pytest.approx(0.2)
    assert report.executed_full_l1 == pytest.approx(0.2)
    assert report.executed_half_l1 == pytest.approx(0.1)
    assert report.executed_cost == pytest.approx(0.002)


def test_rebalance_turnover_report_rejects_cost_without_execution() -> None:
    with pytest.raises(ValueError, match="requires executed_trade_weights"):
        build_rebalance_turnover_report(
            previous_holdings=None,
            target_holdings=["NEW"],
            previous_target_weights=None,
            target_weights=pd.Series({"NEW": 1.0}),
            pretrade_trade_weights=pd.Series({"NEW": 1.0}),
            executed_cost=0.001,
        )
