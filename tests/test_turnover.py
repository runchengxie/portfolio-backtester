from __future__ import annotations

import pandas as pd
import pytest

from cstree.backtesting.turnover import (
    annualize_turnover,
    name_turnover,
    turnover_from_trade_weights,
)


def test_turnover_breakdown_preserves_initial_charge_convention() -> None:
    breakdown = turnover_from_trade_weights(
        pd.Series({"000001.SZ": 0.6, "000002.SZ": 0.4}),
        is_initial=True,
    )

    assert breakdown.buy_weight == pytest.approx(1.0)
    assert breakdown.sell_weight == pytest.approx(0.0)
    assert breakdown.gross_traded_weight == pytest.approx(1.0)
    assert breakdown.half_l1_turnover == pytest.approx(0.5)
    assert breakdown.one_way_turnover == pytest.approx(1.0)


def test_turnover_breakdown_exposes_buy_sell_and_gross_weights() -> None:
    breakdown = turnover_from_trade_weights(
        pd.Series({"kept": 0.0, "sold": -0.75, "bought": 0.75})
    )

    assert breakdown.buy_weight == pytest.approx(0.75)
    assert breakdown.sell_weight == pytest.approx(0.75)
    assert breakdown.gross_traded_weight == pytest.approx(1.5)
    assert breakdown.half_l1_turnover == pytest.approx(0.75)
    assert breakdown.one_way_turnover == pytest.approx(0.75)


def test_turnover_breakdown_keeps_cash_change_visible() -> None:
    breakdown = turnover_from_trade_weights(pd.Series({"position": -0.2}))

    assert breakdown.sell_weight == pytest.approx(0.2)
    assert breakdown.gross_traded_weight == pytest.approx(0.2)
    assert breakdown.half_l1_turnover == pytest.approx(0.1)
    assert breakdown.one_way_turnover == pytest.approx(0.1)


def test_name_turnover_is_distinct_from_weight_turnover() -> None:
    assert name_turnover(
        {"A", "B", "C", "D"},
        {"A", "E", "F", "G"},
    ) == pytest.approx(0.75)
    assert name_turnover(None, {"A", "B"}) == pytest.approx(1.0)
    assert name_turnover(set(), set()) == pytest.approx(0.0)


def test_annualize_turnover_is_linear() -> None:
    assert annualize_turnover(0.75) == pytest.approx(189.0)
