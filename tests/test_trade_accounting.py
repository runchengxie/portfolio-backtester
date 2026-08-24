from __future__ import annotations

from typing import cast

import pandas as pd
import pytest

from portfolio_backtester.trade_accounting import compute_trade_summary, drift_previous_weights


def _timestamp(value: str) -> pd.Timestamp:
    return cast(pd.Timestamp, pd.Timestamp(value))


def _price_table() -> pd.DataFrame:
    return pd.DataFrame(
        {"A": [10.0, 20.0], "B": [10.0, 10.0]},
        index=pd.to_datetime(["2026-01-05", "2026-01-06"]),
    )


def test_drift_previous_weights_reflects_price_moves_before_rebalance() -> None:
    weights = pd.Series({"A": 0.5, "B": 0.5})
    prices = pd.Series({"A": 10.0, "B": 10.0})

    drifted = drift_previous_weights(
        weights,
        prices,
        _timestamp("2026-01-05"),
        _timestamp("2026-01-06"),
        price_table=_price_table(),
    )

    assert drifted["A"] == pytest.approx(2 / 3)
    assert drifted["B"] == pytest.approx(1 / 3)
    assert float(drifted.sum()) == pytest.approx(1.0)


def test_trade_summary_accounts_for_drift_before_target_turnover() -> None:
    previous = pd.Series({"A": 0.5, "B": 0.5})
    previous_prices = pd.Series({"A": 10.0, "B": 10.0})
    target = pd.Series({"A": 0.5, "B": 0.5})

    turnover, entry, exit_, trades = compute_trade_summary(
        previous,
        previous_prices,
        _timestamp("2026-01-05"),
        target,
        _timestamp("2026-01-06"),
        price_table=_price_table(),
    )

    assert turnover == pytest.approx(1 / 6)
    assert entry == pytest.approx(1 / 6)
    assert exit_ == pytest.approx(1 / 6)
    assert trades["A"] == pytest.approx(-1 / 6)
    assert trades["B"] == pytest.approx(1 / 6)


def test_first_portfolio_counts_full_target_as_entry_turnover() -> None:
    target = pd.Series({"A": 0.6, "B": 0.4})

    turnover, entry, exit_, trades = compute_trade_summary(
        None,
        None,
        None,
        target,
        _timestamp("2026-01-06"),
        price_table=_price_table(),
    )

    assert turnover == pytest.approx(1.0)
    assert entry == pytest.approx(1.0)
    assert exit_ == pytest.approx(0.0)
    assert trades.to_dict() == pytest.approx({"A": 0.6, "B": 0.4})
