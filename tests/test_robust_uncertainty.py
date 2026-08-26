from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_backtester import (
    add_conservative_score,
    box_worst_case_return,
    conservative_score,
)


def test_conservative_score_zero_uncertainty_is_identity() -> None:
    score = np.array([0.10, -0.02, 0.03])
    uncertainty = np.zeros(3)

    actual = conservative_score(score, uncertainty, aversion=2.0)

    np.testing.assert_allclose(actual, score)


def test_conservative_score_penalizes_uncertain_candidates() -> None:
    score = np.array([0.10, 0.10, 0.10])
    uncertainty = np.array([0.00, 0.01, 0.03])

    actual = conservative_score(score, uncertainty, aversion=2.0)

    np.testing.assert_allclose(actual, [0.10, 0.08, 0.04])


def test_conservative_score_rejects_negative_or_non_finite_inputs() -> None:
    with pytest.raises(ValueError, match="uncertainty"):
        conservative_score([0.1], [-0.01])
    with pytest.raises(ValueError, match="finite"):
        conservative_score([np.nan], [0.01])
    with pytest.raises(ValueError, match="aversion"):
        conservative_score([0.1], [0.01], aversion=-1.0)


def test_conservative_score_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same shape"):
        conservative_score([0.1, 0.2], [0.01])


def test_add_conservative_score_preserves_rows_and_inputs() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["A", "B"],
            "signal": [0.05, 0.04],
            "signal_uncertainty": [0.01, 0.03],
        },
        index=[7, 3],
    )

    actual = add_conservative_score(
        frame,
        score_col="signal",
        uncertainty_col="signal_uncertainty",
        output_col="signal_robust",
        aversion=1.0,
    )

    assert actual is not frame
    assert actual.index.tolist() == [7, 3]
    assert frame.columns.tolist() == ["symbol", "signal", "signal_uncertainty"]
    np.testing.assert_allclose(actual["signal_robust"], [0.04, 0.01])


def test_box_worst_case_return_penalizes_absolute_exposure() -> None:
    weights = np.array([0.6, -0.4])
    expected_returns = np.array([0.08, -0.03])
    uncertainty_radius = np.array([0.02, 0.01])

    actual = box_worst_case_return(weights, expected_returns, uncertainty_radius)

    nominal = 0.6 * 0.08 + (-0.4) * (-0.03)
    penalty = 0.6 * 0.02 + 0.4 * 0.01
    assert actual == pytest.approx(nominal - penalty)


def test_box_worst_case_return_validates_inputs() -> None:
    with pytest.raises(ValueError, match="same shape"):
        box_worst_case_return([1.0], [0.1, 0.2], [0.01])
    with pytest.raises(ValueError, match="uncertainty"):
        box_worst_case_return([1.0], [0.1], [-0.01])
    with pytest.raises(ValueError, match="finite"):
        box_worst_case_return([1.0], [np.inf], [0.01])
