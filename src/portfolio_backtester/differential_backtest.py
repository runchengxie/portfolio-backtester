"""Differential diagnostics for two canonical backtest backend results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .backends import CanonicalBacktestResult

DIFFERENTIAL_BACKTEST_SCHEMA = "differential_backtest_report.v1"


@dataclass(frozen=True)
class DifferentialBacktestReport:
    reference_backend: str
    candidate_backend: str
    performance_differences: pd.DataFrame
    position_differences: pd.DataFrame
    daily_ledger_differences: pd.DataFrame
    summary: dict[str, Any]
    schema_version: str = DIFFERENTIAL_BACKTEST_SCHEMA

    def describe(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "reference_backend": self.reference_backend,
            "candidate_backend": self.candidate_backend,
            "summary": self.summary,
        }


def _assert_unique_comparison_keys(
    frame: pd.DataFrame,
    *,
    keys: tuple[str, ...],
    label: str,
) -> None:
    if frame.empty:
        return
    missing = [key for key in keys if key not in frame.columns]
    if missing:
        return
    if frame.duplicated(subset=list(keys), keep=False).any():
        raise ValueError(
            f"{label} must have unique comparison keys: " + ", ".join(keys)
        )


def _metric_columns(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    keys: tuple[str, ...],
    preferred: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    shared = [
        column
        for column in reference.columns
        if column in candidate.columns and column not in keys
    ]
    if preferred is not None:
        shared = [column for column in preferred if column in shared]
    metrics: list[str] = []
    for column in shared:
        left = pd.to_numeric(reference[column], errors="coerce")
        right = pd.to_numeric(candidate[column], errors="coerce")
        if left.notna().any() or right.notna().any():
            metrics.append(column)
    return tuple(metrics)


def _numeric_differences(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    keys: tuple[str, ...],
    metrics: tuple[str, ...],
    tolerance: float,
    label: str,
) -> pd.DataFrame:
    columns = [*keys, "metric", "reference", "candidate", "delta"]
    if not metrics:
        return pd.DataFrame(columns=columns)
    if reference.empty and candidate.empty:
        return pd.DataFrame(columns=columns)

    _assert_unique_comparison_keys(reference, keys=keys, label=f"reference {label}")
    _assert_unique_comparison_keys(candidate, keys=keys, label=f"candidate {label}")
    left = (
        reference.loc[:, [*keys, *metrics]].copy()
        if not reference.empty
        else pd.DataFrame(columns=[*keys, *metrics])
    )
    right = (
        candidate.loc[:, [*keys, *metrics]].copy()
        if not candidate.empty
        else pd.DataFrame(columns=[*keys, *metrics])
    )
    merged = left.merge(
        right,
        on=list(keys),
        how="outer",
        suffixes=("__reference", "__candidate"),
        validate="one_to_one",
    )

    rows: list[dict[str, Any]] = []
    for metric in metrics:
        left_values = pd.to_numeric(merged[f"{metric}__reference"], errors="coerce")
        right_values = pd.to_numeric(merged[f"{metric}__candidate"], errors="coerce")
        comparable = left_values.notna() & right_values.notna()
        delta = right_values - left_values
        different = (~comparable) | (delta.abs() > tolerance)
        for index in merged.index[different]:
            row: dict[str, Any] = {key: merged.at[index, key] for key in keys}
            row.update(
                {
                    "metric": metric,
                    "reference": (
                        float(left_values.at[index])
                        if pd.notna(left_values.at[index])
                        else np.nan
                    ),
                    "candidate": (
                        float(right_values.at[index])
                        if pd.notna(right_values.at[index])
                        else np.nan
                    ),
                    "delta": float(delta.at[index]) if pd.notna(delta.at[index]) else np.nan,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def _difference_summary(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"different_cells": 0, "max_abs_delta": 0.0}
    deltas = pd.to_numeric(frame["delta"], errors="coerce").abs().dropna()
    maximum = float(deltas.max()) if not deltas.empty else None
    return {
        "different_cells": int(len(frame)),
        "max_abs_delta": round(maximum, 12) if maximum is not None else None,
    }


def _capability_differences(
    reference: CanonicalBacktestResult,
    candidate: CanonicalBacktestResult,
) -> dict[str, list[Any]]:
    left = reference.capabilities.to_mapping()
    right = candidate.capabilities.to_mapping()
    return {
        key: [left.get(key), right.get(key)]
        for key in sorted(set(left) | set(right))
        if left.get(key) != right.get(key)
    }


def compare_backtest_results(
    reference: CanonicalBacktestResult,
    candidate: CanonicalBacktestResult,
    *,
    tolerance: float = 1e-8,
) -> DifferentialBacktestReport:
    """Compare two normalized backend results and localize numeric differences.

    This function intentionally operates after backend normalization. An RQAlpha,
    Qlib, LEAN, Backtrader, or other adapter first translates its native result
    into ``CanonicalBacktestResult``; differential analysis then remains stable
    and independent of third-party object models.
    """

    if tolerance < 0:
        raise ValueError("tolerance must be >= 0")
    reference.validate()
    candidate.validate()

    comparison_specs = (
        ("performance", reference.performance, candidate.performance, ("period_end",)),
        (
            "positions",
            reference.positions,
            candidate.positions,
            ("rebalance_date", "symbol"),
        ),
        ("daily_ledger", reference.daily_ledger, candidate.daily_ledger, ("trade_date",)),
    )
    for label, left_frame, right_frame, keys in comparison_specs:
        _assert_unique_comparison_keys(left_frame, keys=keys, label=f"reference {label}")
        _assert_unique_comparison_keys(right_frame, keys=keys, label=f"candidate {label}")

    performance_metrics = _metric_columns(
        reference.performance,
        candidate.performance,
        keys=("period_end",),
    )
    performance_differences = _numeric_differences(
        reference.performance,
        candidate.performance,
        keys=("period_end",),
        metrics=performance_metrics,
        tolerance=tolerance,
        label="performance",
    )

    position_metrics = _metric_columns(
        reference.positions,
        candidate.positions,
        keys=("rebalance_date", "symbol"),
        preferred=("weight",),
    )
    position_differences = _numeric_differences(
        reference.positions,
        candidate.positions,
        keys=("rebalance_date", "symbol"),
        metrics=position_metrics,
        tolerance=tolerance,
        label="positions",
    )

    ledger_metrics = _metric_columns(
        reference.daily_ledger,
        candidate.daily_ledger,
        keys=("trade_date",),
        preferred=("cash", "positions_value", "nav"),
    )
    daily_ledger_differences = _numeric_differences(
        reference.daily_ledger,
        candidate.daily_ledger,
        keys=("trade_date",),
        metrics=ledger_metrics,
        tolerance=tolerance,
        label="daily_ledger",
    )

    row_count_delta = {
        name: int(candidate.frames()[name].shape[0] - reference.frames()[name].shape[0])
        for name in reference.frames()
    }
    summary = {
        "performance": _difference_summary(performance_differences),
        "positions": _difference_summary(position_differences),
        "daily_ledger": _difference_summary(daily_ledger_differences),
        "row_count_delta": row_count_delta,
        "capability_differences": _capability_differences(reference, candidate),
    }
    return DifferentialBacktestReport(
        reference_backend=reference.backend_name,
        candidate_backend=candidate.backend_name,
        performance_differences=performance_differences,
        position_differences=position_differences,
        daily_ledger_differences=daily_ledger_differences,
        summary=summary,
    )


__all__ = [
    "DIFFERENTIAL_BACKTEST_SCHEMA",
    "DifferentialBacktestReport",
    "compare_backtest_results",
]
