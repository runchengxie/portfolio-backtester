from __future__ import annotations

import math

import pandas as pd
import pytest

from portfolio_backtester.sharpe_inference import (
    annualized_sharpe_to_periodic,
    annualized_variance_to_periodic,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
    sharpe_standard_error,
)
from portfolio_backtester.strategy_risk import (
    probabilistic_sharpe_ratio as probabilistic_sharpe_ratio_from_returns,
)


def test_sharpe_frequency_conversions() -> None:
    assert annualized_sharpe_to_periodic(1.2, 12.0) == pytest.approx(1.2 / math.sqrt(12.0))
    assert annualized_variance_to_periodic(0.24, 12.0) == pytest.approx(0.02)


def test_probabilistic_and_deflated_sharpe_are_probabilities() -> None:
    se = sharpe_standard_error(
        sharpe=0.2,
        periods=60,
        skew=0.1,
        kurtosis_excess=0.5,
    )
    assert se > 0
    psr = probabilistic_sharpe_ratio(
        sharpe=0.2,
        benchmark_sharpe=0.0,
        periods=60,
        skew=0.1,
        kurtosis_excess=0.5,
    )
    expected_max = expected_max_sharpe(n_trials=20, var_sharpe=0.01)
    dsr, returned_expected_max = deflated_sharpe_ratio(
        sharpe=0.2,
        periods=60,
        skew=0.1,
        kurtosis_excess=0.5,
        n_trials=20,
        var_sharpe=0.01,
    )
    assert 0.0 <= psr <= 1.0
    assert expected_max > 0
    assert returned_expected_max == pytest.approx(expected_max)
    assert 0.0 <= dsr <= 1.0


def test_return_based_psr_delegates_to_the_same_statistical_core() -> None:
    returns = pd.Series([0.01, -0.02, 0.03, 0.015, -0.005, 0.02])
    periods_per_year = 12.0
    benchmark_annualized = 0.4
    periodic_sharpe = float(returns.mean() / returns.std(ddof=1))
    expected = probabilistic_sharpe_ratio(
        sharpe=periodic_sharpe,
        benchmark_sharpe=benchmark_annualized / math.sqrt(periods_per_year),
        periods=len(returns),
        skew=float(returns.skew()),
        kurtosis_excess=float(returns.kurtosis()),
    )

    actual = probabilistic_sharpe_ratio_from_returns(
        returns,
        benchmark_sharpe=benchmark_annualized,
        periods_per_year=periods_per_year,
    )

    assert actual == pytest.approx(expected)
