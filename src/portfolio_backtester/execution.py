"""Execution assumptions (entry/exit, costs, slippage, constraints).

Public symbols are re-exported from the private submodules:

* :mod:`portfolio_backtester._execution_models`
* :mod:`portfolio_backtester._execution_build`

The split is behavior-preserving; external imports from this module are unchanged.
"""

from __future__ import annotations

from ._execution_build import (
    build_cost_model as build_cost_model,
    build_entry_policy as build_entry_policy,
    build_execution_model as build_execution_model,
    build_exit_policy as build_exit_policy,
    build_selection_constraints as build_selection_constraints,
    build_slippage_model as build_slippage_model,
    describe_cost_model as describe_cost_model,
    describe_execution_model as describe_execution_model,
    describe_selection_constraints as describe_selection_constraints,
    describe_slippage_model as describe_slippage_model,
    l2_price_tiered_slippage as l2_price_tiered_slippage,
    required_pricing_columns as required_pricing_columns,
)
from ._execution_models import (
    BpsCostModel as BpsCostModel,
    BpsSlippageModel as BpsSlippageModel,
    CostModel as CostModel,
    DetailedTradeFeeModel as DetailedTradeFeeModel,
    EntryPolicy as EntryPolicy,
    ExecutionModel as ExecutionModel,
    ExitFallbackPolicy as ExitFallbackPolicy,
    ExitPolicy as ExitPolicy,
    ExitPricePolicy as ExitPricePolicy,
    NoCostModel as NoCostModel,
    NoSlippageModel as NoSlippageModel,
    ParticipationSlippageModel as ParticipationSlippageModel,
    SelectionConstraints as SelectionConstraints,
    SideBpsCostModel as SideBpsCostModel,
    SlippageModel as SlippageModel,
)

__all__ = [
    "BpsCostModel",
    "BpsSlippageModel",
    "CostModel",
    "DetailedTradeFeeModel",
    "EntryPolicy",
    "ExecutionModel",
    "ExitFallbackPolicy",
    "ExitPolicy",
    "ExitPricePolicy",
    "NoCostModel",
    "NoSlippageModel",
    "ParticipationSlippageModel",
    "SelectionConstraints",
    "SideBpsCostModel",
    "SlippageModel",
    "build_cost_model",
    "build_entry_policy",
    "build_execution_model",
    "build_exit_policy",
    "build_selection_constraints",
    "build_slippage_model",
    "describe_cost_model",
    "describe_execution_model",
    "describe_selection_constraints",
    "describe_slippage_model",
    "l2_price_tiered_slippage",
    "required_pricing_columns",
]
