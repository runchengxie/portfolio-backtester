"""Small, explicit robustness primitives for portfolio construction.

This module deliberately does not estimate uncertainty and does not solve an
optimization problem.  Callers supply a score/return estimate and a
non-negative uncertainty radius derived from strictly out-of-sample evidence.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

FloatVector = Sequence[float] | np.ndarray


def _finite_vector(values: FloatVector, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _uncertainty_vector(values: FloatVector, *, name: str = "uncertainty") -> np.ndarray:
    array = _finite_vector(values, name=name)
    if np.any(array < 0.0):
        raise ValueError(f"{name} must be non-negative")
    return array


def conservative_score(
    score: FloatVector,
    uncertainty: FloatVector,
    *,
    aversion: float = 1.0,
) -> np.ndarray:
    """Return the lower-bound score under independent box uncertainty.

    The transform is ``score - aversion * uncertainty``.  It is a deterministic
    pre-processing primitive, not an uncertainty estimator or optimizer.
    """

    score_array = _finite_vector(score, name="score")
    uncertainty_array = _uncertainty_vector(uncertainty)
    if score_array.shape != uncertainty_array.shape:
        raise ValueError("score and uncertainty must have the same shape")
    if not np.isfinite(aversion) or aversion < 0.0:
        raise ValueError("aversion must be finite and non-negative")
    return score_array - float(aversion) * uncertainty_array


def add_conservative_score(
    frame: pd.DataFrame,
    *,
    score_col: str,
    uncertainty_col: str,
    output_col: str = "conservative_score",
    aversion: float = 1.0,
) -> pd.DataFrame:
    """Return a copy of ``frame`` with an uncertainty-penalized score column.

    Row order and index are preserved.  Missing input columns raise ``KeyError``
    instead of silently synthesizing uncertainty.
    """

    missing = [column for column in (score_col, uncertainty_col) if column not in frame.columns]
    if missing:
        raise KeyError(f"missing columns: {missing}")
    result = frame.copy()
    result[output_col] = conservative_score(
        frame[score_col].to_numpy(dtype=float),
        frame[uncertainty_col].to_numpy(dtype=float),
        aversion=aversion,
    )
    return result


def box_worst_case_return(
    weights: FloatVector,
    expected_returns: FloatVector,
    uncertainty_radius: FloatVector,
) -> float:
    """Return worst-case linear portfolio return under box uncertainty.

    For independent return intervals ``mu_i +/- radius_i``, the exact worst
    case of ``w @ mu`` is ``w @ mu - sum(abs(w_i) * radius_i)``.  Absolute
    exposure matters, so the expression is valid for both long and short
    positions.
    """

    weight_array = _finite_vector(weights, name="weights")
    return_array = _finite_vector(expected_returns, name="expected_returns")
    radius_array = _uncertainty_vector(uncertainty_radius, name="uncertainty_radius")
    if not (weight_array.shape == return_array.shape == radius_array.shape):
        raise ValueError(
            "weights, expected_returns and uncertainty_radius must have the same shape"
        )
    nominal = float(np.dot(weight_array, return_array))
    penalty = float(np.dot(np.abs(weight_array), radius_array))
    return nominal - penalty
