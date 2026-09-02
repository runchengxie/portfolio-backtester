from __future__ import annotations

import pandas as pd
import pytest

from portfolio_backtester.factor_attribution import (
    attribute_factor_return,
    attribute_factor_risk,
)


def _inputs():
    weights = pd.Series({"A": 0.6, "B": 0.4})
    benchmark = pd.Series({"A": 0.5, "B": 0.5})
    exposures = pd.DataFrame(
        {"value": [1.0, 0.0], "momentum": [0.0, 1.0]},
        index=["A", "B"],
    )
    return weights, benchmark, exposures


def test_return_attribution_reconciles_net_active_return() -> None:
    weights, benchmark, exposures = _inputs()

    report = attribute_factor_return(
        weights=weights,
        benchmark_weights=benchmark,
        exposures=exposures,
        factor_returns=pd.Series({"value": 0.02, "momentum": -0.01}),
        portfolio_return=0.012,
        benchmark_return=0.010,
        transaction_cost=0.0005,
    )

    assert report.active_return == pytest.approx(0.002)
    assert report.factor_contributions.to_dict() == pytest.approx(
        {"value": 0.002, "momentum": 0.001}
    )
    assert report.specific_return == pytest.approx(-0.0005)
    assert report.cost_contribution == pytest.approx(-0.0005)
    assert report.reconciled_active_return == pytest.approx(report.active_return)


def test_risk_attribution_reconciles_active_variance() -> None:
    weights, benchmark, exposures = _inputs()

    report = attribute_factor_risk(
        weights=weights,
        benchmark_weights=benchmark,
        exposures=exposures,
        factor_covariance=pd.DataFrame(
            [[0.04, 0.0], [0.0, 0.01]],
            index=["value", "momentum"],
            columns=["value", "momentum"],
        ),
        specific_risk=pd.Series({"A": 0.10, "B": 0.20}),
    )

    assert report.factor_variance_contributions.sum() + report.specific_variance_contribution == (
        pytest.approx(report.active_variance)
    )
    assert report.active_variance > 0
    assert report.factor_variance_contributions["value"] > 0
    assert report.factor_variance_contributions["momentum"] > 0


def test_attribution_rejects_asset_or_factor_mismatch() -> None:
    weights, benchmark, exposures = _inputs()

    with pytest.raises(ValueError, match="assets"):
        attribute_factor_return(
            weights=weights,
            benchmark_weights=benchmark.drop("B"),
            exposures=exposures,
            factor_returns=pd.Series({"value": 0.02, "momentum": -0.01}),
            portfolio_return=0.01,
            benchmark_return=0.0,
        )

    with pytest.raises(ValueError, match="factors"):
        attribute_factor_risk(
            weights=weights,
            benchmark_weights=benchmark,
            exposures=exposures,
            factor_covariance=pd.DataFrame([[0.04]], index=["value"], columns=["value"]),
            specific_risk=pd.Series({"A": 0.10, "B": 0.20}),
        )


def test_risk_attribution_rejects_indefinite_factor_covariance() -> None:
    weights, benchmark, exposures = _inputs()

    with pytest.raises(ValueError, match="positive semidefinite"):
        attribute_factor_risk(
            weights=weights,
            benchmark_weights=benchmark,
            exposures=exposures,
            factor_covariance=pd.DataFrame(
                [[0.01, 0.02], [0.02, 0.01]],
                index=["value", "momentum"],
                columns=["value", "momentum"],
            ),
            specific_risk=pd.Series({"A": 0.10, "B": 0.20}),
        )
