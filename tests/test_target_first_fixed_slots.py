from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from portfolio_backtester import BacktestSpec, StrategySpec, backtest_topk
from portfolio_backtester.execution import build_execution_model
from portfolio_backtester.portfolio import build_positions_by_rebalance
from portfolio_backtester.portfolio_selection import apply_rebalance_buffer


def _position_frame(*, tradable_a: bool = True) -> pd.DataFrame:
    date = pd.Timestamp("2024-01-02")
    return pd.DataFrame(
        {
            "trade_date": [date] * 4,
            "symbol": ["A", "B", "C", "D"],
            "score": [4.0, 3.0, 2.0, 1.0],
            "close": [10.0] * 4,
            "tradable": [tradable_a, True, True, True],
        }
    )


def test_fixed_slots_and_strict_entry_cutoff_leave_unfilled_slots_in_cash() -> None:
    date = pd.Timestamp("2024-01-02")
    data = _position_frame()

    fixed = build_positions_by_rebalance(
        data,
        pred_col="score",
        price_col="close",
        rebalance_dates=[date],
        top_k=4,
        shift_days=0,
        entry_dates_by_rebalance={date: date},
        entry_rank_cutoff=2,
        target_weight_policy="fixed_slot",
    )
    legacy = build_positions_by_rebalance(
        data,
        pred_col="score",
        price_col="close",
        rebalance_dates=[date],
        top_k=4,
        shift_days=0,
        entry_dates_by_rebalance={date: date},
    )

    assert fixed["symbol"].tolist() == ["A", "B"]
    assert fixed["weight"].tolist() == pytest.approx([0.25, 0.25])
    assert fixed["weight"].sum() == pytest.approx(0.5)
    assert legacy["symbol"].tolist() == ["A", "B", "C", "D"]
    assert legacy["weight"].sum() == pytest.approx(1.0)


def test_strict_entry_cutoff_keeps_buffered_incumbents_without_weak_name_fallback() -> None:
    selected = apply_rebalance_buffer(
        ["A", "B", "C", "D", "E", "F"],
        prev_holdings={"D", "E"},
        k=4,
        buffer_exit=2,
        buffer_entry=0,
        entry_rank_cutoff=1,
    )

    assert selected == ["D", "E", "A"]


def test_target_first_freezes_names_before_entry_tradability_is_applied() -> None:
    date = pd.Timestamp("2024-01-02")
    data = _position_frame(tradable_a=False)
    common = {
        "pred_col": "score",
        "price_col": "close",
        "rebalance_dates": [date],
        "top_k": 2,
        "shift_days": 0,
        "entry_dates_by_rebalance": {date: date},
        "tradable_col": "tradable",
    }

    execution_aware = build_positions_by_rebalance(
        data,
        **common,
        selection_price_policy="execution_aware",
    )
    target_first = build_positions_by_rebalance(
        data,
        **common,
        selection_price_policy="target_first",
    )

    assert execution_aware["symbol"].tolist() == ["B", "C"]
    assert target_first["symbol"].tolist() == ["A", "B"]


def test_backtest_reports_frozen_target_separately_from_modeled_entry() -> None:
    signal_dates = pd.to_datetime(["2024-01-02", "2024-01-04"])
    signal_data = pd.DataFrame(
        [
            {"trade_date": date, "symbol": symbol, "score": score}
            for date in signal_dates
            for symbol, score in zip(
                ["A", "B", "C", "D"],
                [4.0, 3.0, 2.0, 1.0],
                strict=True,
            )
        ]
    )
    pricing_dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    pricing_data = pd.DataFrame(
        [
            {
                "trade_date": date,
                "symbol": symbol,
                "close": 12.0 if date == pricing_dates[-1] and symbol == "B" else 10.0,
                "tradable": not (date == pricing_dates[0] and symbol == "A"),
            }
            for date in pricing_dates
            for symbol in ["A", "B", "C", "D"]
        ]
    )

    result = backtest_topk(
        signal_data,
        pred_col="score",
        price_col="close",
        rebalance_dates=list(signal_dates),
        top_k=4,
        shift_days=0,
        cost_bps=10.0,
        trading_days_per_year=252,
        pricing_data=pricing_data,
        tradable_col="tradable",
        entry_rank_cutoff=2,
        selection_price_policy="target_first",
        target_weight_policy="fixed_slot",
    )

    assert result is not None
    stats, net, gross, turnover, periods = result
    assert gross.tolist() == pytest.approx([0.05])
    assert net.tolist() == pytest.approx([0.04975])
    assert turnover.tolist() == pytest.approx([0.25])
    period = periods[0]
    assert period["target_entered_names"] == ("A", "B")
    assert period["target_weight_full_l1"] == pytest.approx(0.5)
    assert period["target_gross_exposure"] == pytest.approx(0.5)
    assert period["target_cash_weight"] == pytest.approx(0.5)
    assert period["pretrade_demand_buy"] == pytest.approx(0.25)
    assert period["modeled_gross_exposure"] == pytest.approx(0.25)
    assert period["modeled_cash_weight"] == pytest.approx(0.75)
    assert period["modeled_total_cost"] == pytest.approx(0.00025)
    assert stats["avg_target_cash_weight"] == pytest.approx(0.5)
    assert stats["avg_modeled_cash_weight"] == pytest.approx(0.75)

    normalized_result = backtest_topk(
        signal_data,
        pred_col="score",
        price_col="close",
        rebalance_dates=list(signal_dates),
        top_k=4,
        shift_days=0,
        cost_bps=0.0,
        trading_days_per_year=252,
        pricing_data=pricing_data,
        tradable_col="tradable",
        entry_rank_cutoff=2,
        selection_price_policy="target_first",
    )
    assert normalized_result is not None
    normalized_period = normalized_result[-1][0]
    assert normalized_period["target_gross_exposure"] == pytest.approx(1.0)
    assert normalized_period["modeled_gross_exposure"] == pytest.approx(0.5)
    assert normalized_period["modeled_cash_weight"] == pytest.approx(0.5)


def _backtest_spec() -> BacktestSpec:
    execution = build_execution_model(
        {},
        default_cost_bps=0.0,
        default_exit_price_policy="strict",
        default_exit_fallback_policy="ffill",
        default_price_col="close",
    )
    return BacktestSpec(
        strategy=StrategySpec(
            name="fixed-slot",
            type="topk_buffered_long_only",
            score_col="score",
            top_k=10,
            weighting="equal",
            long_only=True,
        ),
        execution=execution,
        rebalance_dates=(pd.Timestamp("2024-01-02"),),
        shift_days=1,
        trading_days_per_year=252,
    )


def test_backtest_spec_round_trips_target_and_execution_policies() -> None:
    spec = replace(
        _backtest_spec(),
        entry_rank_cutoff=8,
        selection_price_policy="target_first",
        target_weight_policy="fixed_slot",
    )

    restored = BacktestSpec.from_mapping(spec.to_mapping())
    legacy_mapping = spec.to_mapping()
    for field in ("entry_rank_cutoff", "selection_price_policy", "target_weight_policy"):
        legacy_mapping.pop(field)
    legacy = BacktestSpec.from_mapping(legacy_mapping)

    assert restored == spec
    assert legacy.entry_rank_cutoff is None
    assert legacy.selection_price_policy == "execution_aware"
    assert legacy.target_weight_policy == "normalized"


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"entry_rank_cutoff": 0}, "entry_rank_cutoff must be > 0"),
        ({"selection_price_policy": "unknown"}, "selection_price_policy"),
        ({"target_weight_policy": "unknown"}, "target_weight_policy"),
    ],
)
def test_backtest_spec_rejects_invalid_target_controls(
    changes: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(_backtest_spec(), **changes)
