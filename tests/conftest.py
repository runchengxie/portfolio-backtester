from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]

sys.path[:0] = [
    str(ROOT / "src"),
]


@pytest.fixture
def a_share_replay_request():
    from cstree.backtesting.backends import PositionReplayRequest
    from cstree.backtesting.position_backtest import PositionBacktestConfig

    positions = pd.DataFrame(
        [
            {"rebalance_date": "20200101", "symbol": "AAA", "weight": 0.5},
            {"rebalance_date": "20200101", "symbol": "BBB", "weight": 0.5},
        ]
    )
    pricing = pd.DataFrame(
        [
            {"trade_date": "20200102", "symbol": "AAA", "close": 10.0, "tradable": True},
            {"trade_date": "20200102", "symbol": "BBB", "close": 20.0, "tradable": True},
            {"trade_date": "20200103", "symbol": "AAA", "close": 11.0, "tradable": False},
            {"trade_date": "20200103", "symbol": "BBB", "close": 18.0, "tradable": True},
            {"trade_date": "20200104", "symbol": "AAA", "close": 12.0, "tradable": True},
            {"trade_date": "20200104", "symbol": "BBB", "close": 22.0, "tradable": True},
        ]
    )
    periods = pd.DataFrame(
        [
            {
                "rebalance_date": "20200101",
                "entry_idx": 0,
                "planned_exit_idx": 1,
                "exit_idx": 2,
                "entry_date": "20200102",
                "planned_exit_date": "20200103",
                "exit_date": "20200104",
            }
        ]
    )
    return PositionReplayRequest(
        positions=positions,
        pricing=pricing,
        periods=periods,
        config=PositionBacktestConfig(
            transaction_cost_bps=10.0,
            exit_price_policy="delay",
            tradable_col="tradable",
        ),
    )
