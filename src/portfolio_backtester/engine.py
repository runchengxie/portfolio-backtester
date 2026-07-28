"""Top-K backtest engine.

Public and internal helpers are re-exported here; their definitions live in the
private submodules :mod:`_engine_leg` and :mod:`_engine_periods`. Original
behavior is unchanged. ``backtest_topk`` is re-exported from :mod:`api`, and
``_compute_trade_summary`` from :mod:`leg_helpers`, matching the prior layout.
"""

from __future__ import annotations

from ._engine_periods import _run_backtest_config as _run_backtest_config
from .api import backtest_topk as backtest_topk
from .leg_helpers import _compute_trade_summary as _compute_trade_summary

__all__ = [
    "_compute_trade_summary",
    "_run_backtest_config",
    "backtest_topk",
]
