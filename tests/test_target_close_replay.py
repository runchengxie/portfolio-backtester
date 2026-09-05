import pandas as pd
import pytest

from portfolio_backtester import target_close_replay as replay


def test_missing_formation_is_rejected_instead_of_silently_dropped():
    prices = pd.DataFrame({"A": [10.0, 10.0]}, index=pd.date_range("2020-01-01", periods=2))
    targets = pd.DataFrame({"formation_date": [pd.NaT], "symbol": ["A"], "weight": [1.0]})
    with pytest.raises(ValueError, match="target keys"):
        replay.replay_close_targets(targets, prices)


def test_sell_block_defers_trade_but_marks_held_stock_at_observed_price():
    dates = pd.date_range("2020-01-01", periods=4)
    prices = pd.DataFrame({"A": [10.0, 10.0, 8.0, 9.0]}, index=dates)
    targets = pd.DataFrame(
        {"formation_date": dates[:2], "symbol": ["A", "A"], "weight": [1.0, 0.0]}
    )
    blocked = pd.DataFrame({"A": [False, False, True, False]}, index=dates)
    result = replay.replay_close_targets(targets, prices, cost_bps=0, sell_blocked=blocked)
    assert result.nav.tolist() == pytest.approx([1.0, 1.0, 0.8, 0.9])
    assert result.rebalance_deferred.tolist() == [False, False, True, False]
    assert result.sell_notional.tolist() == pytest.approx([0.0, 0.0, 0.0, 0.9])


def test_buy_block_does_not_prevent_sale_in_opposite_direction():
    dates = pd.date_range("2020-01-01", periods=4)
    prices = pd.DataFrame({"A": [10.0, 10.0, 11.0, 12.0]}, index=dates)
    targets = pd.DataFrame(
        {"formation_date": dates[:3], "symbol": ["A"] * 3, "weight": [1.0, 1.0, 0.0]}
    )
    blocked = pd.DataFrame({"A": [False, True, False, True]}, index=dates)
    result = replay.replay_close_targets(targets, prices, cost_bps=0, buy_blocked=blocked)
    assert result.buy_notional.tolist() == pytest.approx([0.0, 0.0, 1.0, 0.0])
    assert result.sell_notional.iloc[-1] == pytest.approx(12 / 11)


def test_next_close_execution_excludes_pre_entry_return_and_drifts():
    prices = pd.DataFrame(
        {"A": [10.0, 20.0, 40.0, 20.0], "B": [10.0] * 4},
        index=pd.date_range("2020-01-01", periods=4),
    )
    targets = pd.DataFrame(
        {"formation_date": [prices.index[0]] * 2, "symbol": ["A", "B"], "weight": [0.5, 0.5]}
    )
    result = replay.replay_close_targets(targets, prices, cost_bps=0)
    assert result.net_return.tolist() == pytest.approx([0.0, 0.0, 0.5, -1 / 3])
    assert result.nav.iloc[-1] == pytest.approx(1.0)


def test_repeated_exposure_changes_charge_both_sales_and_repurchases():
    dates = pd.date_range("2020-01-01", periods=4)
    prices = pd.DataFrame({"A": [10.0] * 4}, index=dates)
    targets = pd.DataFrame(
        {"formation_date": dates[:3], "symbol": ["A"] * 3, "weight": [1.0, 0.5, 1.0]}
    )
    result = replay.replay_close_targets(targets, prices, cost_bps=100)
    assert result.sell_notional.iloc[2] > 0
    assert result.buy_notional.iloc[3] > 0
    assert result.cost.iloc[2] > 0 and result.cost.iloc[3] > 0
    assert result.nav.iloc[-1] < result.nav.iloc[1]
    assert result.nav.iloc[-1] == pytest.approx(1 - result.cost.sum())


def test_initial_buy_is_self_financing_after_cost():
    prices = pd.DataFrame({"A": [10.0, 10.0]}, index=pd.date_range("2020-01-01", periods=2))
    targets = pd.DataFrame({"formation_date": [prices.index[0]], "symbol": ["A"], "weight": [1.0]})
    result = replay.replay_close_targets(targets, prices, cost_bps=100)
    assert result.nav.iloc[-1] == pytest.approx(1 / 1.01)
    assert result.cost.sum() == pytest.approx(0.01 / 1.01)


def test_holdings_log_records_actual_drift_and_reconciles_nav():
    dates = pd.date_range("2020-01-01", periods=3)
    prices = pd.DataFrame({"A": [10.0, 10.0, 20.0], "B": [10.0] * 3}, index=dates)
    targets = pd.DataFrame(
        {
            "formation_date": [dates[0]] * 2,
            "symbol": ["A", "B"],
            "weight": [0.5, 0.5],
        }
    )
    log = []
    daily = replay.replay_close_targets(targets, prices, cost_bps=0, holdings_log=log)
    held = pd.DataFrame(log)
    last = held.loc[held.trade_date.eq(dates[-1])].set_index("symbol")
    assert last.loc["A", "weight"] == pytest.approx(2 / 3)
    assert last.loc["B", "weight"] == pytest.approx(1 / 3)
    assert last.market_value.sum() == pytest.approx(daily.nav.iloc[-1])
    assert not held.trade_date.eq(dates[0]).any()


def test_exposure_order_executes_next_close_without_resetting_stock_weights():
    dates = pd.date_range("2020-01-01", periods=5)
    prices = pd.DataFrame({"A": [10.0, 10.0, 20.0, 20.0, 40.0], "B": [10.0] * 5}, index=dates)
    targets = pd.DataFrame(
        {"formation_date": [dates[0]] * 2, "symbol": ["A", "B"], "weight": [0.5, 0.5]}
    )
    exposure = pd.Series([0.5], index=[dates[2]])
    result = replay.replay_close_targets(targets, prices, cost_bps=0, exposure=exposure)
    assert result.sell_notional.tolist() == pytest.approx([0.0, 0.0, 0.0, 0.75, 0.0])
    # Half of drifted A=1/B=.5 is retained: A=.5, B=.25, cash=.75.
    assert result.nav.iloc[-1] == pytest.approx(2.0)


def test_exposure_orders_pay_sale_and_reentry_costs():
    dates = pd.date_range("2020-01-01", periods=4)
    prices = pd.DataFrame({"A": [10.0] * 4}, index=dates)
    targets = pd.DataFrame({"formation_date": [dates[0]], "symbol": ["A"], "weight": [1.0]})
    exposure = pd.Series([0.5, 1.0], index=dates[1:3])
    result = replay.replay_close_targets(targets, prices, cost_bps=100, exposure=exposure)
    assert result.sell_notional.iloc[2] > 0
    assert result.buy_notional.iloc[3] > 0
    assert result.cost.iloc[2] > 0 and result.cost.iloc[3] > 0
    assert result.nav.iloc[-1] == pytest.approx(1 - result.cost.sum())


def test_invalid_exposure_is_rejected():
    dates = pd.date_range("2020-01-01", periods=2)
    prices = pd.DataFrame({"A": [10.0] * 2}, index=dates)
    targets = pd.DataFrame({"formation_date": [dates[0]], "symbol": ["A"], "weight": [1.0]})
    with pytest.raises(ValueError, match="exposure"):
        replay.replay_close_targets(targets, prices, exposure=pd.Series([1.1], index=dates[:1]))


def test_delayed_exposure_reduction_preserves_weights_at_actual_execution():
    dates = pd.date_range("2020-01-01", periods=6)
    prices = pd.DataFrame({"A": [10.0, 10.0, 20.0, 20.0, 40.0, 80.0], "B": [10.0] * 6}, index=dates)
    halted = pd.DataFrame(False, index=dates, columns=prices.columns)
    halted.loc[dates[3], "B"] = True
    targets = pd.DataFrame(
        {"formation_date": [dates[0]] * 2, "symbol": ["A", "B"], "weight": [0.5, 0.5]}
    )
    result = replay.replay_close_targets(
        targets,
        prices,
        cost_bps=0,
        suspended=halted,
        exposure=pd.Series([0.5], index=[dates[2]]),
    )
    assert result.rebalance_deferred.iloc[3]
    assert result.sell_notional.iloc[4] == pytest.approx(1.25)
    assert result.nav.iloc[-1] == pytest.approx(3.5)


def test_rebalance_trades_against_drifted_holdings():
    dates = pd.date_range("2020-01-01", periods=4)
    prices = pd.DataFrame({"A": [10.0, 10.0, 20.0, 20.0], "B": [10.0] * 4}, index=dates)
    targets = pd.DataFrame(
        {
            "formation_date": [dates[0]] * 2 + [dates[2]] * 2,
            "symbol": ["A", "B"] * 2,
            "weight": [0.5] * 4,
        }
    )
    result = replay.replay_close_targets(targets, prices, cost_bps=0)
    assert result.nav.iloc[-1] == pytest.approx(1.5)
    assert result.buy_notional.iloc[-1] == pytest.approx(0.25)
    assert result.sell_notional.iloc[-1] == pytest.approx(0.25)
    assert result.traded_fraction.iloc[-1] == pytest.approx(1 / 3)


def test_cash_reduction_pays_sale_cost():
    dates = pd.date_range("2020-01-01", periods=3)
    prices = pd.DataFrame({"A": [10.0] * 3}, index=dates)
    targets = pd.DataFrame(
        {"formation_date": [dates[0], dates[1]], "symbol": ["A", "A"], "weight": [1.0, 0.0]}
    )
    result = replay.replay_close_targets(targets, prices, cost_bps=100)
    assert result.nav.iloc[-1] == pytest.approx(0.99 / 1.01)
    assert result.sell_notional.iloc[-1] == pytest.approx(1 / 1.01)


def test_missing_marks_are_rejected():
    prices = pd.DataFrame({"A": [10.0, float("nan")]}, index=pd.date_range("2020-01-01", periods=2))
    targets = pd.DataFrame({"formation_date": [prices.index[0]], "symbol": ["A"], "weight": [1.0]})
    with pytest.raises(ValueError, match="complete"):
        replay.replay_close_targets(targets, prices)


def test_missing_marks_before_entry_do_not_remove_newly_listed_stock():
    dates = pd.date_range("2020-01-01", periods=3)
    prices = pd.DataFrame({"A": [float("nan"), 10.0, 20.0]}, index=dates)
    targets = pd.DataFrame({"formation_date": [dates[0]], "symbol": ["A"], "weight": [1.0]})
    result = replay.replay_close_targets(targets, prices, cost_bps=0)
    assert result.nav.iloc[-1] == pytest.approx(2.0)


def test_missing_mark_during_holding_is_not_silently_flat():
    dates = pd.date_range("2020-01-01", periods=3)
    prices = pd.DataFrame({"A": [10.0, 10.0, float("nan")]}, index=dates)
    targets = pd.DataFrame({"formation_date": [dates[0]], "symbol": ["A"], "weight": [1.0]})
    with pytest.raises(ValueError, match="complete"):
        replay.replay_close_targets(targets, prices, cost_bps=0)


def test_confirmed_suspension_delays_exit_and_preserves_reopening_loss():
    dates = pd.date_range("2020-01-01", periods=5)
    prices = pd.DataFrame({"A": [10.0, 10.0, float("nan"), float("nan"), 5.0]}, index=dates)
    suspended = pd.DataFrame({"A": [False, False, True, True, False]}, index=dates)
    targets = pd.DataFrame(
        {"formation_date": [dates[0], dates[1]], "symbol": ["A", "A"], "weight": [1.0, 0.0]}
    )
    result = replay.replay_close_targets(targets, prices, cost_bps=0, suspended=suspended)
    assert result.nav.tolist() == pytest.approx([1.0, 1.0, 1.0, 1.0, 0.5])
    assert result.sell_notional.tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.5])
    assert result.rebalance_deferred.tolist() == [False, False, True, True, False]
