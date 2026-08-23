from __future__ import annotations

import pandas as pd

from portfolio_backtester.style_factors_backtest import available_factor_names


def test_available_factor_names_discovers_public_fund_signals() -> None:
    factors = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2025-08-29", "2025-08-29"]),
            "symbol": ["000001.SZ", "000002.SZ"],
            "factor_fund_breadth_z": [-1.0, 1.0],
            "factor_fund_breadth_change_z": [-0.5, 0.5],
            "factor_fund_ownership_z": [-0.8, 0.8],
            "factor_fund_ownership_change_z": [-0.3, 0.3],
        }
    )

    assert available_factor_names(factors) == [
        "fund_breadth",
        "fund_breadth_change",
        "fund_ownership",
        "fund_ownership_change",
    ]
