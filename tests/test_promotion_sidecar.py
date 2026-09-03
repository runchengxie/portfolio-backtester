import pandas as pd
import pytest

from portfolio_backtester.promotion_sidecar import (
    PromotionSidecarConfig,
    build_promotion_sidecar_config,
    simulate_promotion_sidecar,
)


def test_promotion_sidecar_disabled_and_capacity_partial_fill() -> None:
    assert build_promotion_sidecar_config(None).enabled is False
    positions = pd.DataFrame(
        {
            "rebalance_date": ["2026-01-05"],
            "entry_date": ["2026-01-06"],
            "symbol": ["600519.SH"],
            "weight": [1.0],
        }
    )
    pricing = pd.DataFrame(
        {
            "trade_date": ["2026-01-06"],
            "symbol": ["600519.SH"],
            "close": [10.0],
            "amount": [1000.0],
            "is_buy_tradable": [True],
            "is_sell_tradable": [True],
        }
    )

    result = simulate_promotion_sidecar(
        positions,
        pricing,
        PromotionSidecarConfig(enabled=True, portfolio_value=10_000.0, participation_rate=0.1),
    )

    assert result["summary"]["status"] == "ok"
    assert result["summary"]["fill_ratio"] < 1.0
    assert result["fills"]["unfilled_value"].iloc[0] == pytest.approx(9900.0)
