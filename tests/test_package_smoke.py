from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

import portfolio_backtester

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
    "portfolio_backtester.period_turnover",
    "portfolio_backtester.portfolio_position_options",
    "portfolio_backtester.portfolio_weights",
    "portfolio_backtester.liquidity_proxy",
    "portfolio_backtester.rebalance",
    "portfolio_backtester.position_backtest",
    "portfolio_backtester.position_evaluation",
    "portfolio_backtester.post_buffer_exposure_repair",
    "portfolio_backtester.sharpe_inference",
    "portfolio_backtester.strategy",
    "portfolio_backtester.strategy_risk",
    "portfolio_backtester.turnover",
    "portfolio_backtester.turnover_attribution",
    "portfolio_backtester.types",
)
FORBIDDEN_RUNTIME_PREFIXES = ("alpha_research", "strategy_pipeline.pipeline")


def test_portfolio_backtester_package_uses_owner_native_root() -> None:
    package_root = Path(portfolio_backtester.__file__).resolve().parent
    expected_package_root = (Path(__file__).parents[1] / "src" / "portfolio_backtester").resolve()

    assert package_root == expected_package_root


@pytest.mark.parametrize("module_name", OWNED_MODULES)
def test_owned_modules_import(module_name: str) -> None:
    module = importlib.import_module(module_name)

    assert module.__name__ == module_name


def test_portfolio_backtester_package_exports_core_entrypoints() -> None:
    assert set(portfolio_backtester.__all__) == {
        "BacktestSpec",
        "CostBreakdown",
        "DailyWatch20Config",
        "DailyWatch20Receipt",
        "DailyWatch20Result",
        "DailyWatch20SelectionError",
        "DetailedTradeFeeModel",
        "GroupCap",
        "GuardFactorSpec",
        "HrpConfig",
        "HrpResult",
        "POSITIONS_BY_REBALANCE_CONTRACT",
        "PositionBacktestConfig",
        "PositionBacktestEvaluation",
        "PositionBacktestResult",
        "PositionsByRebalanceFrameContract",
        "RebalanceTurnoverReport",
        "SessionRebalanceSchedule",
        "SizingConfig",
        "StrategyRiskReport",
        "StrategySpec",
        "TurnoverBreakdown",
        "annualize_turnover",
        "annualized_sharpe_to_periodic",
        "annualized_variance_to_periodic",
        "assert_positions_by_rebalance_frame",
        "average_active_bets",
        "backtest_topk",
        "build_portfolio_sizing_receipt",
        "build_rebalance_turnover_report",
        "build_sized_weights",
        "build_sizing_receipt",
        "construct_positions_from_strategy",
        "deflated_sharpe_ratio",
        "discretize_weights",
        "evaluate_position_backtest",
        "expected_max_sharpe",
        "hierarchical_risk_parity",
        "get_session_interval_rebalance_dates",
        "implementation_shortfall_metrics",
        "l2_price_tiered_slippage",
        "name_turnover",
        "probabilistic_sharpe_ratio",
        "probabilistic_sharpe_ratio_from_stats",
        "probability_to_size",
        "return_concentration",
        "rolling_hrp_weights",
        "run_position_backtest",
        "run_backtest",
        "select_daily_watch20",
        "series_sha256",
        "sha256_file",
        "sharpe_standard_error",
        "strategy_failure_probability",
        "strategy_from_config",
        "summarize_period_returns",
        "summarize_strategy_risk",
        "turnover_from_trade_weights",
        "validate_positions_by_rebalance_frame",
        "write_receipt",
    }


def test_owned_modules_do_not_load_sibling_namespaces() -> None:
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
