from __future__ import annotations

import pandas as pd

from portfolio_backtester.portfolio import build_positions_by_rebalance


def test_build_positions_by_rebalance_limits_target_weight_turnover() -> None:
    frame = pd.DataFrame(
        [
            {"trade_date": "2024-01-01", "symbol": "AAA", "pred": 3.0, "close": 10.0},
            {"trade_date": "2024-01-01", "symbol": "BBB", "pred": 2.0, "close": 10.0},
            {"trade_date": "2024-01-01", "symbol": "CCC", "pred": 1.0, "close": 10.0},
            {"trade_date": "2024-01-08", "symbol": "AAA", "pred": 1.0, "close": 10.0},
            {"trade_date": "2024-01-08", "symbol": "BBB", "pred": 2.0, "close": 10.0},
            {"trade_date": "2024-01-08", "symbol": "CCC", "pred": 3.0, "close": 10.0},
        ]
    )

    positions = build_positions_by_rebalance(
        frame,
        pred_col="pred",
        price_col="close",
        rebalance_dates=[pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-08")],
        top_k=2,
        shift_days=0,
        weighting="equal",
        max_turnover_per_rebalance=0.50,
    )

    second = positions.loc[positions["rebalance_date"].eq("20240108")].set_index("symbol")
    assert second["weight"].to_dict() == {"AAA": 0.25, "BBB": 0.5, "CCC": 0.25}
    assert second["rank"].to_dict() == {"AAA": 3, "BBB": 2, "CCC": 1}
