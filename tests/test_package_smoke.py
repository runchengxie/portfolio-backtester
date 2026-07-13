from __future__ import annotations

import importlib
import importlib.metadata
import subprocess
import sys
from pathlib import Path

import pytest

import cstree
from cstree import backtesting

CORE_MODULES = (
    "cstree.backtesting.api",
    "cstree.backtesting.backends",
    "cstree.backtesting.backtest_spec",
    "cstree.backtesting.engine",
    "cstree.backtesting.metrics",
    "cstree.backtesting.parity",
    "cstree.backtesting.integrations",
    "cstree.backtesting.integrations.lean",
    "cstree.backtesting.integrations.qlib",
    "cstree.backtesting.a_share_executable_oos_topk",
    "cstree.backtesting.execution",
    "cstree.backtesting.execution_sim",
    "cstree.backtesting.a_share_round_lot_diagnostics",
    "cstree.backtesting.benchmark_ladder",
    "cstree.backtesting.contracts",
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
    "cstree.backtesting.turnover",
    "cstree.backtesting.turnover_attribution",
    "cstree.backtesting.types",
)
PRODUCT_MODULES = (
    "cstree.backtesting.products",
    "cstree.backtesting.products.daily_watch20",
)
OWNED_MODULES = (*CORE_MODULES, *PRODUCT_MODULES)
FORBIDDEN_RUNTIME_PREFIXES = ("cstree.alpha", "cstree.pipeline")
FORBIDDEN_ML_PREFIXES = ("sklearn", "xgboost")
FORBIDDEN_FRAMEWORK_PREFIXES = ("qlib", "QuantConnect", "AlgorithmImports")


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
        "BacktestBackend",
        "BacktestBackendResult",
        "BacktestDifferentialReport",
        "CostBreakdown",
        "DailyWatch20Config",
        "DailyWatch20Receipt",
        "DailyWatch20Result",
        "DailyWatch20SelectionError",
        "DetailedTradeFeeModel",
        "DifferenceDimension",
        "DifferenceExplanation",
        "GroupCap",
        "GuardFactorSpec",
        "NativeAShareReplayBackend",
        "ParityTolerance",
        "POSITIONS_BY_REBALANCE_CONTRACT",
        "PositionBacktestConfig",
        "PositionBacktestResult",
        "PositionReplayRequest",
        "PositionsByRebalanceFrameContract",
        "StrategySpec",
        "TurnoverBreakdown",
        "annualize_turnover",
        "assert_positions_by_rebalance_frame",
        "backtest_topk",
        "compare_backtest_results",
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


def test_core_import_does_not_load_product_or_ml_modules() -> None:
    code = f"""
import importlib
import sys

for module_name in {CORE_MODULES!r}:
    importlib.import_module(module_name)

for prefix in {FORBIDDEN_ML_PREFIXES!r}:
    offenders = [
        module_name
        for module_name in sys.modules
        if module_name == prefix or module_name.startswith(prefix + ".")
    ]
    if offenders:
        raise SystemExit("loaded ML module(s): " + ", ".join(sorted(offenders)))

for prefix in {FORBIDDEN_FRAMEWORK_PREFIXES!r}:
    offenders = [
        module_name
        for module_name in sys.modules
        if module_name == prefix or module_name.startswith(prefix + ".")
    ]
    if offenders:
        raise SystemExit("loaded framework module(s): " + ", ".join(sorted(offenders)))

product_offenders = [
    module_name
    for module_name in sys.modules
    if module_name == "cstree.backtesting.products"
    or module_name.startswith("cstree.backtesting.products.")
    or module_name == "cstree.backtesting.daily_watch20"
]
if product_offenders:
    raise SystemExit("loaded product module(s): " + ", ".join(sorted(product_offenders)))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr + result.stdout


def test_installed_base_metadata_does_not_require_ml_frameworks() -> None:
    requirements = importlib.metadata.requires("portfolio-backtester") or []
    normalized = [requirement.lower() for requirement in requirements]

    assert not any(requirement.startswith("scikit-learn") for requirement in normalized)
    assert not any(requirement.startswith("xgboost") for requirement in normalized)
    assert any(requirement.startswith("scipy") for requirement in normalized)


def test_top_level_product_exports_are_lazy_compatibility_aliases() -> None:
    code = """
import sys
import cstree.backtesting as backtesting

assert "cstree.backtesting.products" not in sys.modules
from cstree.backtesting.products import DailyWatch20Config as canonical
assert backtesting.DailyWatch20Config is canonical
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr + result.stdout


def test_legacy_daily_watch20_module_reexports_with_deprecation_warning() -> None:
    code = """
import warnings

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always", DeprecationWarning)
    from cstree.backtesting.daily_watch20 import DailyWatch20Config as legacy

from cstree.backtesting.products import DailyWatch20Config as canonical
assert legacy is canonical
messages = [str(item.message) for item in caught if item.category is DeprecationWarning]
assert messages == [
    "cstree.backtesting.daily_watch20 is deprecated; "
    "import DailyWatch20 APIs from cstree.backtesting.products instead"
]
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr + result.stdout
