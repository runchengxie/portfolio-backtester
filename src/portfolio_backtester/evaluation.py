"""Backtest evaluation orchestration and output recording.

Public and internal helpers are re-exported here; their definitions live in the
private submodules :mod:`_evaluation_backtest` and :mod:`_evaluation_positions`.
Original behavior is unchanged, and every symbol (including the ``_``-prefixed
internal helpers imported by other modules and repos) remains reachable from
this path.
"""

from __future__ import annotations

from ._evaluation_backtest import (
    _build_period_positions as _build_period_positions,
    _evaluate_walk_forward_backtest as _evaluate_walk_forward_backtest,
    _record_backtest_outputs as _record_backtest_outputs,
    _record_exposure_outputs as _record_exposure_outputs,
    _record_period_backtest_outputs as _record_period_backtest_outputs,
    _run_period_backtest as _run_period_backtest,
    _run_walk_forward_backtest_topk as _run_walk_forward_backtest_topk,
    _score_walk_forward_backtest_frame as _score_walk_forward_backtest_frame,
    _summarize_walk_forward_benchmark as _summarize_walk_forward_benchmark,
)
from ._evaluation_positions import (
    _execution_trade_fee_model as _execution_trade_fee_model,
    _filter_positions_to_backtest_periods as _filter_positions_to_backtest_periods,
    _rebalance_key as _rebalance_key,
    _record_period_execution_sim as _record_period_execution_sim,
    _record_period_ideal_daily_nav as _record_period_ideal_daily_nav,
)

# Public owner API: cross-repo callers must import the non-underscore name.
evaluate_walk_forward_backtest = _evaluate_walk_forward_backtest

__all__ = [
    "_build_period_positions",
    "_evaluate_walk_forward_backtest",
    "_execution_trade_fee_model",
    "_filter_positions_to_backtest_periods",
    "_rebalance_key",
    "_record_backtest_outputs",
    "_record_exposure_outputs",
    "_record_period_backtest_outputs",
    "_record_period_execution_sim",
    "_record_period_ideal_daily_nav",
    "_run_period_backtest",
    "_run_walk_forward_backtest_topk",
    "_score_walk_forward_backtest_frame",
    "_summarize_walk_forward_benchmark",
    "evaluate_walk_forward_backtest",
]
