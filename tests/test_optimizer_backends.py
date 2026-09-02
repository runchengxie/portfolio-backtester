from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_backtester.optimization import (
    EqualWeightOptimizerBackend,
    HrpOptimizerBackend,
    OptimizerRegistry,
    PortfolioOptimizationRequest,
)


def _returns() -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=80, freq="B")
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "A": rng.normal(0.0005, 0.010, len(index)),
            "B": rng.normal(0.0003, 0.012, len(index)),
            "C": rng.normal(0.0004, 0.009, len(index)),
        },
        index=index,
    )


def test_equal_weight_backend_returns_canonical_weights() -> None:
    request = PortfolioOptimizationRequest(returns=_returns(), max_weight=0.5)

    result = EqualWeightOptimizerBackend().run(request)

    result.validate(request)
    assert result.backend_name == "native.equal_weight"
    assert result.weights.to_dict() == pytest.approx({"A": 1 / 3, "B": 1 / 3, "C": 1 / 3})


def test_hrp_backend_reuses_native_hrp_and_respects_bounds() -> None:
    request = PortfolioOptimizationRequest(
        returns=_returns(),
        min_weight=0.1,
        max_weight=0.6,
        covariance_shrinkage=0.1,
    )

    result = HrpOptimizerBackend().run(request)

    result.validate(request)
    assert result.backend_name == "native.hrp"
    assert result.weights.sum() == pytest.approx(1.0)
    assert result.weights.min() >= 0.1 - 1e-12
    assert result.weights.max() <= 0.6 + 1e-12
    assert result.diagnostics["method"] == "hrp"


def test_hrp_backend_projects_extreme_native_weights_into_box_simplex() -> None:
    index = pd.date_range("2025-01-01", periods=80, freq="B")
    returns = pd.DataFrame(
        {
            "A": np.linspace(-0.001, 0.001, len(index)),
            "B": np.linspace(-0.05, 0.05, len(index)),
            "C": np.linspace(0.04, -0.04, len(index)),
        },
        index=index,
    )
    request = PortfolioOptimizationRequest(
        returns=returns,
        min_weight=0.2,
        max_weight=0.5,
    )

    result = HrpOptimizerBackend().run(request)

    result.validate(request)
    assert result.weights.sum() == pytest.approx(1.0)
    assert result.weights.min() >= 0.2 - 1e-12
    assert result.weights.max() <= 0.5 + 1e-12


def test_single_asset_hrp_keeps_requested_backend_identity() -> None:
    request = PortfolioOptimizationRequest(returns=_returns()[["A"]])

    result = HrpOptimizerBackend().run(request)

    assert result.backend_name == "native.hrp"
    assert result.weights.to_dict() == {"A": 1.0}
    assert result.diagnostics["fallback"] == "single_asset"


def test_optimizer_registry_is_explicit() -> None:
    registry = OptimizerRegistry()
    registry.register(EqualWeightOptimizerBackend())

    result = registry.run("native.equal_weight", PortfolioOptimizationRequest(returns=_returns()))

    assert registry.names() == ("native.equal_weight",)
    assert result.backend_name == "native.equal_weight"
    with pytest.raises(KeyError, match="Unknown optimizer backend"):
        registry.get("missing")


def test_request_rejects_expected_returns_with_wrong_assets() -> None:
    with pytest.raises(ValueError, match="expected_returns"):
        PortfolioOptimizationRequest(
            returns=_returns(),
            expected_returns=pd.Series({"A": 0.01, "B": 0.02}),
        )


def test_request_rejects_infeasible_max_weight() -> None:
    with pytest.raises(ValueError, match="max_weight"):
        PortfolioOptimizationRequest(returns=_returns(), max_weight=0.3)
