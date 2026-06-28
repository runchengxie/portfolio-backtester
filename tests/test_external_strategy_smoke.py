from __future__ import annotations

import pandas as pd
import pytest

from cstree.backtesting import (
    PositionBacktestConfig,
    StrategySpec,
    construct_positions_from_strategy,
    run_position_backtest,
)


def test_external_signal_can_build_positions_and_run_position_backtest() -> None:
    signals = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(
                [
                    "2026-01-05",
                    "2026-01-05",
                    "2026-01-06",
                    "2026-01-06",
                    "2026-01-07",
                    "2026-01-07",
                ]
            ),
            "symbol": ["AAA", "BBB", "AAA", "BBB", "AAA", "BBB"],
            "external_score": [0.9, 0.1, 0.8, 0.2, 0.7, 0.3],
            "close": [10.0, 20.0, 11.0, 19.0, 12.1, 18.0],
        }
    )
    strategy = StrategySpec(
        name="external-momentum",
        type="topk_buffered_long_only",
        score_col="external_score",
        top_k=1,
    )

    positions = construct_positions_from_strategy(
        signals,
        strategy=strategy,
        price_col="close",
        rebalance_dates=[pd.Timestamp("2026-01-05")],
        shift_days=1,
    )
    periods = pd.DataFrame(
        [
            {
                "rebalance_date": "20260105",
                "entry_date": "20260106",
                "exit_date": "20260107",
            }
        ]
    )

    result = run_position_backtest(
        positions=positions,
        pricing=signals[["trade_date", "symbol", "close"]],
        periods=periods,
        config=PositionBacktestConfig(),
    )

    assert positions[["rebalance_date", "entry_date", "symbol", "weight"]].to_dict("records") == [
        {
            "rebalance_date": "20260105",
            "entry_date": "20260106",
            "symbol": "AAA",
            "weight": 1.0,
        }
    ]
    assert result.summary["schema"] == "position_backtest.v1"
    assert result.summary["stats"]["weighting"] == "positions"
    assert result.periods.loc[0, "gross_return"] == pytest.approx(0.10)
