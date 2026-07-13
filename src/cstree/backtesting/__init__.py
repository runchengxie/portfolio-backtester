from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .products import (
        DailyWatch20Config,
        DailyWatch20Receipt,
        DailyWatch20Result,
        DailyWatch20SelectionError,
        GuardFactorSpec,
        select_daily_watch20,
    )

from .api import backtest_topk, run_backtest
from .backtest_spec import BacktestSpec
from .contracts import (
    POSITIONS_BY_REBALANCE_CONTRACT,
    GroupCap,
    PositionsByRebalanceFrameContract,
    StrategySpec,
    assert_positions_by_rebalance_frame,
    validate_positions_by_rebalance_frame,
)
from .execution import DetailedTradeFeeModel, l2_price_tiered_slippage
from .metrics import summarize_period_returns
from .position_backtest import PositionBacktestConfig, PositionBacktestResult, run_position_backtest
from .strategy import construct_positions_from_strategy, strategy_from_config
from .turnover import (
    TurnoverBreakdown,
    annualize_turnover,
    name_turnover,
    turnover_from_trade_weights,
)
from .types import CostBreakdown

_PRODUCT_EXPORTS = frozenset(
    {
        "DailyWatch20Config",
        "DailyWatch20Receipt",
        "DailyWatch20Result",
        "DailyWatch20SelectionError",
        "GuardFactorSpec",
        "select_daily_watch20",
    }
)


def __getattr__(name: str) -> Any:
    if name not in _PRODUCT_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    products = import_module(".products", __name__)
    value = getattr(products, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_PRODUCT_EXPORTS})


__all__ = [
    "POSITIONS_BY_REBALANCE_CONTRACT",
    "BacktestSpec",
    "CostBreakdown",
    "DailyWatch20Config",
    "DailyWatch20Receipt",
    "DailyWatch20Result",
    "DailyWatch20SelectionError",
    "DetailedTradeFeeModel",
    "GroupCap",
    "GuardFactorSpec",
    "PositionBacktestConfig",
    "PositionBacktestResult",
    "PositionsByRebalanceFrameContract",
    "StrategySpec",
    "TurnoverBreakdown",
    "annualize_turnover",
    "assert_positions_by_rebalance_frame",
    "backtest_topk",
    "construct_positions_from_strategy",
    "l2_price_tiered_slippage",
    "name_turnover",
    "run_backtest",
    "run_position_backtest",
    "select_daily_watch20",
    "strategy_from_config",
    "summarize_period_returns",
    "turnover_from_trade_weights",
    "validate_positions_by_rebalance_frame",
]
