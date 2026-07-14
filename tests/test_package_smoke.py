from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

import cstree
from cstree import backtesting

OWNED_MODULES = (
    "portfolio_backtester.api",
    "portfolio_backtester.backtest_spec",
    "portfolio_backtester.engine",
    "portfolio_backtester.metrics",
    "portfolio_backtester.a_share_executable_oos_topk",
    "portfolio_backtester.execution",
    "portfolio_backtester.execution_sim",
    "portfolio_backtester.a_share_round_lot_diagnostics",
    "portfolio_backtester.benchmark_ladder",
    "portfolio_backtester.contracts",
    "portfolio_backtester.daily_watch20",
    "portfolio_backtester.exposure",
    "portfolio_backtester.exposure_screen",
    "portfolio_backtester.reporting",
    "portfolio_backtester.tearsheet",
    "portfolio_backtester.portfolio",
    "portfolio_backtester.portfolio_weights",
    "portfolio_backtester.liquidity_proxy",
    "portfolio_backtester.rebalance",
    "portfolio_backtester.position_backtest",
    "portfolio_backtester.post_buffer_exposure_repair",
    "portfolio_backtester.strategy",
    "portfolio_backtester.turnover",
    "portfolio_backtester.turnover_attribution",
    "portfolio_backtester.types",
)
FORBIDDEN_RUNTIME_PREFIXES = ("alpha_research", "strategy_pipeline.pipeline")


def test_cstree_namespace_includes_backtesting_package_root() -> None:
    namespace_paths = {Path(path).resolve() for path in cstree.__path__}
    package_root = (Path(__file__).parents[1] / "src" / "cstree").resolve()

    assert package_root in namespace_paths


@pytest.mark.parametrize("module_name", OWNED_MODULES)
def test_owned_backtesting_modules_import(module_name: str) -> None:
    module = importlib.import_module(module_name)

    assert module.__name__ == module_name


def test_backtesting_package_exports_core_entrypoints() -> None:
    assert set(backtesting.__all__) == {
        "BacktestSpec",
        "CostBreakdown",
        "DailyWatch20Config",
        "DailyWatch20Receipt",
        "DailyWatch20Result",
        "DailyWatch20SelectionError",
        "DetailedTradeFeeModel",
        "GroupCap",
        "GuardFactorSpec",
        "POSITIONS_BY_REBALANCE_CONTRACT",
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
        "run_position_backtest",
        "run_backtest",
        "select_daily_watch20",
        "strategy_from_config",
        "summarize_period_returns",
        "turnover_from_trade_weights",
        "validate_positions_by_rebalance_frame",
    }


def test_owned_backtesting_modules_do_not_load_alpha_or_pipeline() -> None:
    code = f"""
import importlib
import sys

for module_name in {OWNED_MODULES!r}:
    importlib.import_module(module_name)

for prefix in {FORBIDDEN_RUNTIME_PREFIXES!r}:
    offenders = [
        module_name
        for module_name in sys.modules
        if module_name == prefix or module_name.startswith(prefix + ".")
    ]
    if offenders:
        raise SystemExit("loaded forbidden module(s): " + ", ".join(sorted(offenders)))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr + result.stdout
