from __future__ import annotations

import importlib
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
    "cstree.backtesting.turnover_attribution",
)


def test_cstree_namespace_reaches_workspace_siblings() -> None:
    namespace_paths = {Path(path).as_posix() for path in cstree.__path__}

    assert any(path.endswith("portfolio-backtester/src/cstree") for path in namespace_paths)
    assert any(path.endswith("cross-sectional-trees/src/cstree") for path in namespace_paths)
    assert any(path.endswith("alpha-research/src/cstree") for path in namespace_paths)


@pytest.mark.parametrize("module_name", OWNED_MODULES)
def test_owned_backtesting_modules_import(module_name: str) -> None:
    module = importlib.import_module(module_name)

    assert module.__name__ == module_name


def test_backtesting_package_exports_core_entrypoints() -> None:
    assert set(backtesting.__all__) == {"backtest_topk", "summarize_period_returns"}
