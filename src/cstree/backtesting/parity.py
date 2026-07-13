"""Differential reports for framework-neutral backtest results."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, cast

import numpy as np
import pandas as pd

from .backends import BacktestBackendResult


class DifferenceDimension(StrEnum):
    DATES = "dates"
    POSITIONS = "positions"
    TURNOVER = "turnover"
    COST = "cost"
    PNL = "pnl"


@dataclass(frozen=True)
class DifferenceExplanation:
    """Auditable reason for accepting a semantic backend difference."""

    code: str
    detail: str

    def __post_init__(self) -> None:
        if not self.code.strip() or not self.detail.strip():
            raise ValueError("Difference explanations require a non-empty code and detail.")


@dataclass(frozen=True)
class ParityTolerance:
    position_weight: float = 1e-10
    turnover: float = 1e-10
    cost: float = 1e-10
    pnl: float = 1e-10

    def __post_init__(self) -> None:
        values = (self.position_weight, self.turnover, self.cost, self.pnl)
        if any(value < 0 or not np.isfinite(value) for value in values):
            raise ValueError("Parity tolerances must be finite and non-negative.")


@dataclass(frozen=True)
class DimensionComparison:
    dimension: DifferenceDimension
    matched: bool
    details: dict[str, Any]


@dataclass(frozen=True)
class BacktestDifferentialReport:
    reference_backend: str
    candidate_backend: str
    comparisons: tuple[DimensionComparison, ...]
    explanations: dict[DifferenceDimension, DifferenceExplanation]

    @property
    def differing_dimensions(self) -> tuple[DifferenceDimension, ...]:
        return tuple(item.dimension for item in self.comparisons if not item.matched)

    @property
    def unexplained_dimensions(self) -> tuple[DifferenceDimension, ...]:
        return tuple(
            dimension
            for dimension in self.differing_dimensions
            if dimension not in self.explanations
        )

    @property
    def accepted(self) -> bool:
        """Return true only when every actual difference has an explanation."""

        return not self.unexplained_dimensions

    def to_mapping(self) -> dict[str, Any]:
        comparisons = []
        for comparison in self.comparisons:
            explanation = self.explanations.get(comparison.dimension)
            status = "matched" if comparison.matched else "unexplained"
            if not comparison.matched and explanation is not None:
                status = "explained"
            comparisons.append(
                {
                    "dimension": comparison.dimension.value,
                    "status": status,
                    "details": comparison.details,
                    "explanation": (
                        None
                        if explanation is None
                        else {"code": explanation.code, "detail": explanation.detail}
                    ),
                }
            )
        return {
            "schema": "backtest_differential.v1",
            "reference_backend": self.reference_backend,
            "candidate_backend": self.candidate_backend,
            "accepted": self.accepted,
            "comparisons": comparisons,
        }


def compare_backtest_results(
    reference: BacktestBackendResult,
    candidate: BacktestBackendResult,
    *,
    tolerance: ParityTolerance | None = None,
    explanations: Mapping[DifferenceDimension | str, DifferenceExplanation] | None = None,
) -> BacktestDifferentialReport:
    """Compare dates, positions, turnover, cost and PnL with explicit tolerances."""

    resolved_tolerance = tolerance or ParityTolerance()
    resolved_explanations = _normalize_explanations(explanations)
    comparisons = (
        _compare_dates(reference, candidate),
        _compare_positions(reference, candidate, resolved_tolerance.position_weight),
        _compare_performance_columns(
            DifferenceDimension.TURNOVER,
            reference,
            candidate,
            ("turnover",),
            resolved_tolerance.turnover,
        ),
        _compare_performance_columns(
            DifferenceDimension.COST,
            reference,
            candidate,
            ("fee_cost", "slippage_cost", "total_cost"),
            resolved_tolerance.cost,
        ),
        _compare_performance_columns(
            DifferenceDimension.PNL,
            reference,
            candidate,
            ("gross_return", "net_return", "pnl"),
            resolved_tolerance.pnl,
        ),
    )
    return BacktestDifferentialReport(
        reference.backend,
        candidate.backend,
        comparisons,
        resolved_explanations,
    )


def _normalize_explanations(
    explanations: Mapping[DifferenceDimension | str, DifferenceExplanation] | None,
) -> dict[DifferenceDimension, DifferenceExplanation]:
    if explanations is None:
        return {}
    return {DifferenceDimension(key): value for key, value in explanations.items()}


def _date_strings(values: pd.Series) -> set[str]:
    return {_timestamp_text(value) for value in pd.to_datetime(values)}


def _compare_dates(
    reference: BacktestBackendResult,
    candidate: BacktestBackendResult,
) -> DimensionComparison:
    ref_performance = _date_strings(reference.performance["date"])
    candidate_performance = _date_strings(candidate.performance["date"])
    ref_positions = _date_strings(reference.positions["date"])
    candidate_positions = _date_strings(candidate.positions["date"])
    details = {
        "performance_only_in_reference": sorted(ref_performance - candidate_performance),
        "performance_only_in_candidate": sorted(candidate_performance - ref_performance),
        "positions_only_in_reference": sorted(ref_positions - candidate_positions),
        "positions_only_in_candidate": sorted(candidate_positions - ref_positions),
    }
    return DimensionComparison(
        DifferenceDimension.DATES,
        not any(details.values()),
        details,
    )


def _compare_positions(
    reference: BacktestBackendResult,
    candidate: BacktestBackendResult,
    tolerance: float,
) -> DimensionComparison:
    merged = reference.positions.merge(
        candidate.positions,
        on=["date", "symbol"],
        how="outer",
        suffixes=("_reference", "_candidate"),
        indicator=True,
    )
    missing_reference = merged.loc[merged["_merge"] == "right_only", ["date", "symbol"]]
    missing_candidate = merged.loc[merged["_merge"] == "left_only", ["date", "symbol"]]
    shared = merged.loc[merged["_merge"] == "both"].copy()
    shared["absolute_difference"] = (shared["weight_reference"] - shared["weight_candidate"]).abs()
    mismatched = shared.loc[shared["absolute_difference"] > tolerance]
    details = {
        "only_in_reference": _key_records(missing_candidate),
        "only_in_candidate": _key_records(missing_reference),
        "weight_mismatch_count": int(mismatched.shape[0]),
        "maximum_weight_difference": _maximum(shared["absolute_difference"]),
        "tolerance": tolerance,
    }
    matched = missing_reference.empty and missing_candidate.empty and int(mismatched.shape[0]) == 0
    return DimensionComparison(DifferenceDimension.POSITIONS, matched, details)


def _compare_performance_columns(
    dimension: DifferenceDimension,
    reference: BacktestBackendResult,
    candidate: BacktestBackendResult,
    columns: tuple[str, ...],
    tolerance: float,
) -> DimensionComparison:
    reference_dates = _date_strings(reference.performance["date"])
    candidate_dates = _date_strings(candidate.performance["date"])
    merged = reference.performance[["date", *columns]].merge(
        candidate.performance[["date", *columns]],
        on="date",
        how="inner",
        suffixes=("_reference", "_candidate"),
    )
    column_details: dict[str, Any] = {}
    matched = reference_dates == candidate_dates
    for column in columns:
        differences = (merged[f"{column}_reference"] - merged[f"{column}_candidate"]).abs()
        mismatch_count = int((differences > tolerance).sum())
        matched = matched and mismatch_count == 0
        column_details[column] = {
            "mismatch_count": mismatch_count,
            "maximum_absolute_difference": _maximum(differences),
        }
    details = {
        "overlapping_dates": int(merged.shape[0]),
        "only_in_reference": sorted(reference_dates - candidate_dates),
        "only_in_candidate": sorted(candidate_dates - reference_dates),
        "tolerance": tolerance,
        "columns": column_details,
    }
    return DimensionComparison(dimension, matched, details)


def _maximum(values: pd.Series) -> float:
    return 0.0 if values.empty else float(values.max())


def _key_records(frame: pd.DataFrame) -> list[dict[str, str]]:
    return [
        {
            "date": _timestamp_text(row["date"]),
            "symbol": str(row["symbol"]),
        }
        for row in frame.to_dict("records")
    ]


def _timestamp_text(value: Any) -> str:
    timestamp = cast("pd.Timestamp", pd.Timestamp(value))
    if pd.isna(timestamp):
        raise ValueError("Parity comparison dates must not be missing.")
    return timestamp.isoformat()


__all__ = [
    "BacktestDifferentialReport",
    "DifferenceDimension",
    "DifferenceExplanation",
    "ParityTolerance",
    "compare_backtest_results",
]
