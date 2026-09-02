from __future__ import annotations

import pandas as pd
import pytest

from portfolio_backtester.backends import BackendCapabilities, CanonicalBacktestResult
from portfolio_backtester.differential_backtest import compare_backtest_results


def _result(name: str, *, weight: float = 0.5, nav: float = 100.0) -> CanonicalBacktestResult:
    return CanonicalBacktestResult(
        backend_name=name,
        performance=pd.DataFrame(
            {
                "period_end": ["2026-01-02", "2026-01-05"],
                "return": [0.01, -0.002],
                "nav": [100.0, nav],
            }
        ),
        positions=pd.DataFrame(
            {
                "rebalance_date": ["2026-01-02", "2026-01-02"],
                "symbol": ["000001.SZ", "600000.SH"],
                "weight": [weight, 1.0 - weight],
            }
        ),
        daily_ledger=pd.DataFrame(
            {
                "trade_date": ["2026-01-02", "2026-01-05"],
                "cash": [0.0, 0.0],
                "positions_value": [100.0, nav],
                "nav": [100.0, nav],
            }
        ),
        capabilities=BackendCapabilities(daily_ledger=True, market_rules=("t1", "limit")),
    )


def test_differential_report_localizes_position_and_ledger_differences() -> None:
    report = compare_backtest_results(
        _result("native", weight=0.5, nav=100.0),
        _result("rqalpha", weight=0.52, nav=99.5),
        tolerance=1e-12,
    )

    assert report.reference_backend == "native"
    assert report.candidate_backend == "rqalpha"
    assert not report.position_differences.empty
    assert set(report.position_differences["symbol"]) == {"000001.SZ", "600000.SH"}
    assert report.summary["positions"]["max_abs_delta"] == 0.02
    assert report.summary["daily_ledger"]["max_abs_delta"] == 0.5
    assert report.summary["row_count_delta"]["fills"] == 0


def test_identical_results_have_no_numeric_differences() -> None:
    reference = _result("native")
    candidate = _result("other")

    report = compare_backtest_results(reference, candidate)

    assert report.performance_differences.empty
    assert report.position_differences.empty
    assert report.daily_ledger_differences.empty
    assert report.summary["performance"]["different_cells"] == 0


def test_capability_difference_is_reported() -> None:
    reference = _result("native")
    candidate = CanonicalBacktestResult(
        backend_name="external",
        performance=reference.performance.copy(),
        positions=reference.positions.copy(),
        capabilities=BackendCapabilities(daily_ledger=False, market_rules=("t1",)),
    )

    report = compare_backtest_results(reference, candidate)

    assert report.summary["capability_differences"]["daily_ledger"] == [True, False]
    assert report.summary["capability_differences"]["market_rules"] == [
        ["t1", "limit"],
        ["t1"],
    ]


def test_duplicate_semantic_keys_fail_before_outer_merge() -> None:
    reference = _result("native")
    duplicate = reference.positions.iloc[[0]].copy()
    reference = CanonicalBacktestResult(
        backend_name=reference.backend_name,
        performance=reference.performance,
        positions=pd.concat([reference.positions, duplicate], ignore_index=True),
        daily_ledger=reference.daily_ledger,
        capabilities=reference.capabilities,
    )

    with pytest.raises(ValueError, match="unique comparison keys"):
        compare_backtest_results(reference, _result("other"))
