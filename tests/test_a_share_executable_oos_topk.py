from __future__ import annotations

import pytest

from cstree.backtesting import a_share_executable_oos_topk as executable_topk


def test_flat_cost_mode_preserves_cost_bps_compatibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(executable_topk, "USE_DETAILED_FEES", False)
    monkeypatch.setattr(executable_topk, "COST_BPS", 25.0)

    cost, effective_bps = executable_topk._trade_cost(
        notional=20_000.0,
        delta=100,
        impact_bps=1.5,
    )

    assert cost == 53.0
    assert effective_bps == 26.5


def test_detailed_fee_mode_applies_min_commission_and_sell_taxes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(executable_topk, "USE_DETAILED_FEES", True)
    monkeypatch.setattr(executable_topk, "BUY_COMMISSION_BPS", 2.5)
    monkeypatch.setattr(executable_topk, "SELL_COMMISSION_BPS", 2.5)
    monkeypatch.setattr(executable_topk, "STAMP_TAX_SELL_BPS", 5.0)
    monkeypatch.setattr(executable_topk, "TRANSFER_FEE_BPS", 0.1)
    monkeypatch.setattr(executable_topk, "MIN_COMMISSION_CNY", 5.0)
    monkeypatch.setattr(executable_topk, "BUY_SLIPPAGE_BPS", 10.0)
    monkeypatch.setattr(executable_topk, "SELL_SLIPPAGE_BPS", 10.0)

    buy_cost, buy_bps = executable_topk._trade_cost(
        notional=10_000.0,
        delta=100,
        impact_bps=0.0,
    )
    sell_cost, sell_bps = executable_topk._trade_cost(
        notional=10_000.0,
        delta=-100,
        impact_bps=0.0,
    )

    assert buy_cost == 15.1
    assert round(buy_bps, 2) == 15.10
    assert sell_cost == 20.1
    assert round(sell_bps, 2) == 20.10
