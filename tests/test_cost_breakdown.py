from __future__ import annotations

from typing import cast

import pandas as pd
import pytest

from portfolio_backtester.execution import DetailedTradeFeeModel
from portfolio_backtester.execution_sim import simulate_ideal_daily_nav
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


# --------------------------------------------------------------------------- #
# Stage 3: execution path really splits cost into sub-items
# --------------------------------------------------------------------------- #


def _stage3_sim() -> pd.DataFrame:
    dates = pd.date_range("2020-01-02", periods=4, freq="B")
    # Two entry dates with swapped targets force a sell (AAA) then a buy (BBB),
    # so stamp duty is exercised alongside buy-side costs.
    positions = pd.DataFrame(
        {
            "rebalance_date": ["20200102", "20200102", "20200103", "20200103"],
            "entry_date": ["20200102", "20200102", "20200103", "20200103"],
            "symbol": ["AAA", "BBB", "AAA", "BBB"],
            "weight": [1.0, 0.0, 0.0, 1.0],
            "side": ["long", "long", "long", "long"],
        }
    )
    pricing = pd.DataFrame(
        [
            {
                "trade_date": date,
                "symbol": symbol,
                "open": 10.0 + i,
                "amount": 500_000.0,
                "medadv20_amount": 500_000.0,
                "is_tradable": True,
            }
            for i, date in enumerate(dates)
            for symbol in ["AAA", "BBB"]
        ]
    )
    fee_model = DetailedTradeFeeModel(
        buy_commission_bps=2.5,
        sell_commission_bps=2.5,
        sell_stamp_duty_bps=5.0,
        transfer_fee_bps=0.1,
        min_commission=0.0,
        buy_slippage_bps=6.0,
        sell_slippage_bps=8.0,
    )
    result = simulate_ideal_daily_nav(
        positions,
        pricing,
        price_col="open",
        transaction_cost_bps=0.0,
        portfolio_value=1_000_000.0,
        trade_fee_model=fee_model,
    )
    return result.to_unified_ledger(portfolio_value=1_000_000.0).cost_breakdown


def test_stage3_cost_breakdown_subitems_nonzero_and_conserved() -> None:
    cb = _stage3_sim()
    total = cb.loc[cb["side"].eq("total")].iloc[0]

    # Real split: fee sub-items and spread must be non-zero.
    assert float(total["commission"]) > 0.0
    assert float(total["stamp_tax"]) > 0.0
    assert float(total["transfer_fee"]) > 0.0
    assert float(total["spread_cost"]) > 0.0

    # Impact/opportunity/financing are honest placeholders (no model yet).
    assert float(total["temporary_impact"]) == 0.0
    assert float(total["permanent_impact"]) == 0.0
    assert float(total["opportunity_cost"]) == 0.0
    assert float(total["financing_cost"]) == 0.0

    # Conservation: eight sub-items sum to fee_cost + slippage_cost and to
    # the legacy transaction_cost column.
    subitem_sum = (
        total["commission"]
        + total["stamp_tax"]
        + total["transfer_fee"]
        + total["spread_cost"]
        + total["temporary_impact"]
        + total["permanent_impact"]
        + total["opportunity_cost"]
        + total["financing_cost"]
    )
    assert subitem_sum == pytest.approx(float(total["transaction_cost"]), rel=1e-9)
    assert float(total["fee_cost"]) == pytest.approx(
        float(total["commission"] + total["stamp_tax"] + total["transfer_fee"]), rel=1e-9
    )
    assert float(total["slippage_cost"]) == pytest.approx(float(total["spread_cost"]), rel=1e-9)


def test_stage3_no_fee_model_keeps_single_cost_in_spread() -> None:
    dates = pd.date_range("2020-01-02", periods=3, freq="B")
    positions = pd.DataFrame(
        {
            "rebalance_date": ["20200102", "20200102"],
            "entry_date": ["20200102", "20200102"],
            "symbol": ["AAA", "BBB"],
            "weight": [0.5, 0.5],
            "side": ["long", "long"],
        }
    )
    pricing = pd.DataFrame(
        [
            {
                "trade_date": date,
                "symbol": symbol,
                "open": 10.0,
                "amount": 500_000.0,
                "medadv20_amount": 500_000.0,
                "is_tradable": True,
            }
            for date in dates
            for symbol in ["AAA", "BBB"]
        ]
    )
    result = simulate_ideal_daily_nav(
        positions,
        pricing,
        price_col="open",
        transaction_cost_bps=10.0,
        portfolio_value=1_000_000.0,
    )
    cb = result.to_unified_ledger(portfolio_value=1_000_000.0).cost_breakdown
    total = cb.loc[cb["side"].eq("total")].iloc[0]
    # Legacy single cost_rate path: everything lands in spread_cost, sub-items 0.
    assert float(total["spread_cost"]) == pytest.approx(float(total["transaction_cost"]), rel=1e-9)
    assert float(total["commission"]) == 0.0
    assert float(total["stamp_tax"]) == 0.0
    assert float(total["transfer_fee"]) == 0.0
