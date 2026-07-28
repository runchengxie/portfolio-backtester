"""Position-level backtest from explicit rebalance/period tables.

Public symbols are re-exported from the private submodules:

* :mod:`portfolio_backtester._position_backtest_config`
* :mod:`portfolio_backtester._position_backtest_engine`
* :mod:`portfolio_backtester._position_backtest_cli`

The split is behavior-preserving; external imports from this module are unchanged.
"""

from __future__ import annotations

from ._position_backtest_cli import (
    add_position_backtest_args as add_position_backtest_args,
    run as run,
)
from ._position_backtest_config import (
    PositionBacktestConfig as PositionBacktestConfig,
    PositionBacktestResult as PositionBacktestResult,
    PositionExitPolicy as PositionExitPolicy,
    normalize_position_backtest_periods as normalize_position_backtest_periods,
    normalize_position_backtest_positions as normalize_position_backtest_positions,
    normalize_position_backtest_pricing as normalize_position_backtest_pricing,
)
from ._position_backtest_engine import (
    run_position_backtest as run_position_backtest,
)

__all__ = [
    "PositionBacktestConfig",
    "PositionBacktestResult",
    "PositionExitPolicy",
    "add_position_backtest_args",
    "normalize_position_backtest_periods",
    "normalize_position_backtest_positions",
    "normalize_position_backtest_pricing",
    "run",
    "run_position_backtest",
]
