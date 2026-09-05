import numpy as np
import pandas as pd
import pytest

from portfolio_backtester.holdings_pnl import holding_pnl


def test_previous_units_include_exit_day_but_not_entry_day():
    dates = pd.date_range("2020-01-01", periods=4)
    units = pd.DataFrame({"A": [0.0, 2.0, 2.0, 0.0]}, index=dates)
    prices = pd.DataFrame({"A": [5.0, 10.0, 11.0, 9.0]}, index=dates)
    assert holding_pnl(units, prices).A.tolist() == pytest.approx([0.0, 0.0, 2.0, -4.0])


def test_missing_unheld_prices_allowed_but_exit_mark_required():
    dates = pd.date_range("2020-01-01", periods=4)
    units = pd.DataFrame({"A": [0.0, 2.0, 0.0, 0.0]}, index=dates)
    prices = pd.DataFrame({"A": [np.nan, 10.0, 11.0, np.nan]}, index=dates)
    assert holding_pnl(units, prices).A.tolist() == pytest.approx([0.0, 0.0, 2.0, 0.0])
    prices.iloc[2, 0] = np.nan
    with pytest.raises(ValueError, match="marks"):
        holding_pnl(units, prices)


def test_reject_misaligned_or_invalid_positions():
    dates = pd.date_range("2020-01-01", periods=2)
    units = pd.DataFrame({"A": [0.0, 1.0]}, index=dates)
    with pytest.raises(ValueError, match="aligned"):
        holding_pnl(units, units.rename(columns={"A": "B"}))
    with pytest.raises(ValueError, match="units"):
        holding_pnl(-units, units + 1)
    with pytest.raises(ValueError, match="calendar"):
        holding_pnl(units.iloc[::-1], units.iloc[::-1])


def test_reconciles_costed_ledger_with_exit_and_cash():
    from portfolio_backtester.target_close_replay import replay_close_targets

    dates = pd.date_range("2020-01-01", periods=5)
    prices = pd.DataFrame({"A": [10.0, 10.0, 12.0, 9.0, 8.0]}, index=dates)
    targets = pd.DataFrame(
        {
            "formation_date": [dates[0], dates[2]],
            "symbol": ["A", "A"],
            "weight": [0.5, 0.0],
        }
    )
    log = []
    ledger = replay_close_targets(targets, prices, cost_bps=25, holdings_log=log)
    units = pd.DataFrame(log).pivot(index="trade_date", columns="symbol", values="adjusted_units")
    units = units.reindex(index=dates, columns=prices.columns).fillna(0)
    pnl = holding_pnl(units, prices).sum(axis=1)
    net_change = ledger.nav.diff().fillna(ledger.nav.iloc[0] - 1).to_numpy()
    np.testing.assert_allclose(pnl.to_numpy() - ledger.cost.to_numpy(), net_change, atol=1e-14)
