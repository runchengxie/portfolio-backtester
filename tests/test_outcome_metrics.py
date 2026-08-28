from __future__ import annotations

import math

import pytest

from portfolio_backtester.outcome_metrics import summarize_outcome_distribution


def test_summarize_outcome_distribution_reports_distribution_and_path_metrics() -> None:
    report = summarize_outcome_distribution(
        returns=[-0.20, -0.10, 0.00, 0.10, 0.30],
        mfe=[0.00, 0.02, 0.05, 0.15, 0.40],
        mae=[-0.25, -0.15, -0.05, 0.00, 0.00],
        peak_giveback=[0.05, 0.04, 0.03, 0.10, 0.20],
        holding_periods=[1, 2, 3, 5, 10],
    )

    assert report.observations == 5
    assert report.mean_return == pytest.approx(0.02)
    assert report.median_return == pytest.approx(0.0)
    assert report.loss_probability == pytest.approx(0.4)
    assert report.q05_return == pytest.approx(-0.18)
    assert report.q25_return == pytest.approx(-0.10)
    assert report.q75_return == pytest.approx(0.10)
    assert report.q95_return == pytest.approx(0.26)
    assert report.cvar_05_return == pytest.approx(-0.20)
    assert report.mean_mfe == pytest.approx(0.124)
    assert report.median_mfe == pytest.approx(0.05)
    assert report.mean_mae == pytest.approx(-0.09)
    assert report.median_mae == pytest.approx(-0.05)
    assert report.mean_peak_giveback == pytest.approx(0.084)
    assert report.p90_peak_giveback == pytest.approx(0.16)
    assert report.mean_holding_period == pytest.approx(4.2)
    assert report.median_holding_period == pytest.approx(3.0)
    assert report.p90_holding_period == pytest.approx(8.0)


def test_outcome_report_serialization_is_deterministic() -> None:
    report = summarize_outcome_distribution(
        returns=[-0.1, 0.2],
        mfe=[0.0, 0.3],
        mae=[-0.2, 0.0],
        peak_giveback=[0.1, 0.05],
        holding_periods=[2, 4],
    )

    first = report.to_dict()
    second = report.to_dict()

    assert first == second
    assert first["schema_version"] == "outcome_distribution.v1"
    assert first["observations"] == 2


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {
                "returns": [],
                "mfe": [],
                "mae": [],
                "peak_giveback": [],
                "holding_periods": [],
            },
            "must not be empty",
        ),
        (
            {
                "returns": [0.1, math.nan],
                "mfe": [0.2, 0.3],
                "mae": [-0.1, -0.2],
                "peak_giveback": [0.1, 0.1],
                "holding_periods": [1, 2],
            },
            "finite",
        ),
        (
            {
                "returns": [0.1, 0.2],
                "mfe": [0.2],
                "mae": [-0.1, -0.2],
                "peak_giveback": [0.1, 0.1],
                "holding_periods": [1, 2],
            },
            "same length",
        ),
        (
            {
                "returns": [0.1],
                "mfe": [0.2],
                "mae": [-0.1],
                "peak_giveback": [0.1],
                "holding_periods": [-1],
            },
            "holding_periods",
        ),
        (
            {
                "returns": [0.1],
                "mfe": [-0.01],
                "mae": [-0.1],
                "peak_giveback": [0.1],
                "holding_periods": [1],
            },
            "mfe",
        ),
        (
            {
                "returns": [0.1],
                "mfe": [0.2],
                "mae": [0.01],
                "peak_giveback": [0.1],
                "holding_periods": [1],
            },
            "mae",
        ),
        (
            {
                "returns": [0.1],
                "mfe": [0.2],
                "mae": [-0.1],
                "peak_giveback": [-0.01],
                "holding_periods": [1],
            },
            "peak_giveback",
        ),
    ],
)
def test_outcome_metrics_fail_closed_on_invalid_inputs(
    kwargs: dict[str, list[float]],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        summarize_outcome_distribution(**kwargs)
