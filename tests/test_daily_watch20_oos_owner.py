from __future__ import annotations

import pandas as pd

from portfolio_backtester.daily_watch20_oos import portfolio_daily_rows


def test_equal_weight_daily_rows_are_deterministic() -> None:
    scored = pd.DataFrame(
        {
            "trade_date": [pd.Timestamp("2026-07-15")] * 2,
            "forward_label_start_date": [pd.Timestamp("2026-07-16")] * 2,
            "symbol": ["000001.SZ", "600000.SH"],
            "relative_percentile": [0.9, 0.8],
            "forward_return_1d": [0.01, 0.03],
        }
    )
    pricing = pd.DataFrame(
        {
            "trade_date": [pd.Timestamp("2026-07-16")] * 2,
            "symbol": ["000001.SZ", "600000.SH"],
            "is_suspended": [0, 0],
            "open": [10.0, 10.0],
            "up_limit": [11.0, 11.0],
            "down_limit": [9.0, 9.0],
        }
    )
    result = portfolio_daily_rows(
        scored,
        pricing,
        portfolio_size=2,
        single_side_cost_bps=10,
    )
    assert len(result) == 1
    assert result.loc[0, "gross_forward_return_proxy"] == 0.02
    assert result.loc[0, "tradability_audit_passed"]
