from __future__ import annotations

import pandas as pd

from cstree.backtesting.backends import BacktestBackendResult
from cstree.backtesting.parity import (
    DifferenceDimension,
    DifferenceExplanation,
    ParityTolerance,
    compare_backtest_results,
)


def _result(backend: str = "reference") -> BacktestBackendResult:
    performance = pd.DataFrame(
        {
            "date": ["2020-01-02", "2020-01-03"],
            "gross_return": [0.01, 0.02],
            "net_return": [0.009, 0.018],
            "turnover": [0.3, 0.4],
            "fee_cost": [0.001, 0.002],
            "slippage_cost": [0.0, 0.0],
            "total_cost": [0.001, 0.002],
            "pnl": [0.009, 0.027162],
        }
    )
    positions = pd.DataFrame(
        {
            "date": ["2020-01-02", "2020-01-03"],
            "symbol": ["AAA", "AAA"],
            "weight": [0.5, 1.0],
        }
    )
    return BacktestBackendResult(backend, performance, positions, {})


def test_identical_results_are_accepted_without_explanations() -> None:
    report = compare_backtest_results(_result(), _result("candidate"))

    assert report.accepted is True
    assert report.differing_dimensions == ()
    assert {item["status"] for item in report.to_mapping()["comparisons"]} == {"matched"}


def test_each_semantic_difference_must_be_explained_before_acceptance() -> None:
    reference = _result()
    candidate_performance = reference.performance.copy()
    candidate_performance.loc[0, ["turnover", "fee_cost", "total_cost"]] += 0.01
    candidate_performance.loc[0, ["gross_return", "net_return", "pnl"]] += 0.02
    candidate_performance.loc[1, "date"] = pd.Timestamp("2020-01-04")
    candidate_positions = reference.positions.copy()
    candidate_positions.loc[0, "weight"] = 0.6
    candidate_positions.loc[1, "date"] = pd.Timestamp("2020-01-04")
    candidate = BacktestBackendResult(
        "candidate",
        candidate_performance,
        candidate_positions,
        {},
    )

    unexplained = compare_backtest_results(reference, candidate)

    assert unexplained.accepted is False
    assert set(unexplained.unexplained_dimensions) == set(DifferenceDimension)

    explanations = {
        dimension: DifferenceExplanation(
            code=f"expected_{dimension.value}",
            detail=f"The candidate uses a documented {dimension.value} convention.",
        )
        for dimension in DifferenceDimension
    }
    explained = compare_backtest_results(reference, candidate, explanations=explanations)

    assert explained.accepted is True
    assert explained.unexplained_dimensions == ()
    assert {item["status"] for item in explained.to_mapping()["comparisons"]} == {"explained"}


def test_tolerance_classifies_small_numeric_differences_as_matched() -> None:
    reference = _result()
    candidate_performance = reference.performance.copy()
    candidate_performance.loc[0, "turnover"] += 5e-7
    candidate = BacktestBackendResult(
        "candidate",
        candidate_performance,
        reference.positions,
        {},
    )

    report = compare_backtest_results(
        reference,
        candidate,
        tolerance=ParityTolerance(turnover=1e-6),
    )

    assert report.accepted is True
    turnover = next(
        item for item in report.comparisons if item.dimension is DifferenceDimension.TURNOVER
    )
    assert turnover.matched is True


def test_report_exposes_machine_readable_difference_magnitudes() -> None:
    reference = _result()
    positions = reference.positions.copy()
    positions.loc[0, "weight"] = 0.7
    candidate = BacktestBackendResult("candidate", reference.performance, positions, {})

    report = compare_backtest_results(reference, candidate)
    position_comparison = next(
        item for item in report.comparisons if item.dimension is DifferenceDimension.POSITIONS
    )

    assert position_comparison.details["weight_mismatch_count"] == 1
    assert position_comparison.details["maximum_weight_difference"] == 0.19999999999999996


def test_shifted_dates_make_numeric_dimensions_uncomparable() -> None:
    reference = _result()
    performance = reference.performance.copy()
    performance["date"] = performance["date"] + pd.Timedelta(10, unit="D")
    candidate = BacktestBackendResult("candidate", performance, reference.positions, {})

    report = compare_backtest_results(
        reference,
        candidate,
        explanations={
            DifferenceDimension.DATES: DifferenceExplanation(
                code="calendar_shift",
                detail="Candidate uses a deliberately shifted test calendar.",
            )
        },
    )

    assert report.accepted is False
    assert set(report.unexplained_dimensions) == {
        DifferenceDimension.TURNOVER,
        DifferenceDimension.COST,
        DifferenceDimension.PNL,
    }
