from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from cstree.backtesting import (
    PositionBacktestConfig,
    StrategySpec,
    construct_positions_from_strategy,
    run_position_backtest,
)

ROOT = Path(__file__).resolve().parents[1]


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


def test_external_signal_smoke_runs_without_alpha_or_pipeline_namespace() -> None:
    code = """
import sys

import pandas as pd

import cstree
from cstree.backtesting import (
    PositionBacktestConfig,
    StrategySpec,
    construct_positions_from_strategy,
    run_position_backtest,
)

namespace_paths = [str(path) for path in cstree.__path__]
if any("alpha-research" in path or "strategy-pipeline" in path for path in namespace_paths):
    raise SystemExit("unexpected sibling cstree namespace path(s): " + ", ".join(namespace_paths))

signals = pd.DataFrame(
    {
        "trade_date": pd.to_datetime(
            ["2026-01-05", "2026-01-05", "2026-01-06", "2026-01-06", "2026-01-07", "2026-01-07"]
        ),
        "symbol": ["AAA", "BBB", "AAA", "BBB", "AAA", "BBB"],
        "external_score": [0.9, 0.1, 0.8, 0.2, 0.7, 0.3],
        "close": [10.0, 20.0, 11.0, 19.0, 12.1, 18.0],
    }
)
positions = construct_positions_from_strategy(
    signals,
    strategy=StrategySpec(
        name="external-momentum",
        type="topk_buffered_long_only",
        score_col="external_score",
        top_k=1,
    ),
    price_col="close",
    rebalance_dates=[pd.Timestamp("2026-01-05")],
    shift_days=1,
)
periods = pd.DataFrame(
    [{"rebalance_date": "20260105", "entry_date": "20260106", "exit_date": "20260107"}]
)
result = run_position_backtest(
    positions=positions,
    pricing=signals[["trade_date", "symbol", "close"]],
    periods=periods,
    config=PositionBacktestConfig(),
)
if abs(float(result.periods.loc[0, "gross_return"]) - 0.10) > 1e-12:
    raise SystemExit("unexpected gross return")

offenders = sorted(
    name
    for name in sys.modules
    if name == "cstree.alpha"
    or name.startswith("cstree.alpha.")
    or name == "cstree.pipeline"
    or name.startswith("cstree.pipeline.")
)
if offenders:
    raise SystemExit("loaded forbidden module(s): " + ", ".join(offenders))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.returncode == 0, result.stderr + result.stdout
