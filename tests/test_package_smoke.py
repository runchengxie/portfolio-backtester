from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

import cstree
from cstree import backtesting

OWNED_MODULES = (
    "cstree.backtesting.engine",
    "cstree.backtesting.metrics",
    "cstree.backtesting.a_share_executable_oos_topk",
    "cstree.backtesting.execution",
    "cstree.backtesting.execution_sim",
    "cstree.backtesting.a_share_round_lot_diagnostics",
    "cstree.backtesting.benchmark_ladder",
    "cstree.backtesting.contracts",
    "cstree.backtesting.daily_watch20",
    "cstree.backtesting.exposure",
    "cstree.backtesting.exposure_screen",
    "cstree.backtesting.reporting",
    "cstree.backtesting.tearsheet",
    "cstree.backtesting.portfolio",
    "cstree.backtesting.portfolio_weights",
    "cstree.backtesting.liquidity_proxy",
    "cstree.backtesting.rebalance",
    "cstree.backtesting.position_backtest",
    "cstree.backtesting.post_buffer_exposure_repair",
    "cstree.backtesting.strategy",
    "cstree.backtesting.style_replica_portfolio",
    "cstree.backtesting.turnover_attribution",
)
FORBIDDEN_RUNTIME_PREFIXES = ("cstree.alpha", "cstree.pipeline")


def test_cstree_namespace_includes_backtesting_package_root() -> None:
    namespace_paths = {Path(path).as_posix() for path in cstree.__path__}

    assert any(path.endswith("portfolio-backtester/src/cstree") for path in namespace_paths)


@pytest.mark.parametrize("module_name", OWNED_MODULES)
def test_owned_backtesting_modules_import(module_name: str) -> None:
    module = importlib.import_module(module_name)

    assert module.__name__ == module_name


def test_backtesting_package_exports_core_entrypoints() -> None:
    assert set(backtesting.__all__) == {
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
        "StyleReplicaPortfolioConfig",
        "assert_positions_by_rebalance_frame",
        "backtest_topk",
        "build_style_replica_positions",
        "compute_daily_changes",
        "compute_daily_exposure",
        "compute_style_exposure_summary",
        "construct_positions_from_strategy",
        "l2_price_tiered_slippage",
        "run_position_backtest",
        "select_daily_watch20",
        "strategy_from_config",
        "summarize_period_returns",
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
