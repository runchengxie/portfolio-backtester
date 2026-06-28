from __future__ import annotations

import pandas as pd

from cstree.backtesting.strategy import construct_positions_from_strategy, strategy_from_config


def _signals() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2026-01-05", "2026-01-05", "2026-01-06", "2026-01-06"]),
            "symbol": ["600519.SH", "000858.SZ", "600519.SH", "000858.SZ"],
            "signal_backtest": [0.2, 0.1, 0.3, 0.05],
            "close": [10.0, 11.0, 10.5, 10.8],
        }
    )


def test_strategy_from_config_maps_legacy_backtest_and_constructs_positions() -> None:
    strategy = strategy_from_config(
        {
            "backtest": {
                "top_k": 1,
                "buffer_exit": 0,
                "buffer_entry": 0,
                "weighting": "equal",
            }
        }
    )

    positions = construct_positions_from_strategy(
        _signals(),
        strategy=strategy,
        price_col="close",
        rebalance_dates=[pd.Timestamp("2026-01-05")],
        shift_days=1,
    )

    assert strategy.source == "legacy_backtest_mapping"
    assert positions["symbol"].tolist() == ["600519.SH"]


def test_strategy_from_config_supports_explicit_group_cap_and_execution() -> None:
    strategy = strategy_from_config(
        {
            "strategy": {
                "name": "industry-aware",
                "type": "topk_buffered_long_short",
                "score_col": "signal_backtest",
                "top_k": 20,
                "short_k": 10,
                "buffer_exit": 5,
                "buffer_entry": 2,
                "weighting": "equal",
                "long_only": False,
                "group_cap": {"column": "industry", "max_names": 3},
                "execution": {"cost": {"name": "none"}},
            }
        }
    )

    assert strategy.source == "explicit"
    assert strategy.name == "industry-aware"
    assert strategy.group_cap is not None
    assert strategy.group_cap.column == "industry"
    assert strategy.group_cap.max_names == 3
    assert strategy.execution == {"cost": {"name": "none"}}
