"""Framework-neutral portfolio optimization boundary with native baselines."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from .hrp import HrpConfig, hierarchical_risk_parity

PORTFOLIO_OPTIMIZATION_RESULT_SCHEMA = "portfolio_optimization_result.v1"


def _numeric_series_for_assets(
    value: pd.Series | None,
    assets: tuple[str, ...],
    *,
    label: str,
) -> pd.Series | None:
    if value is None:
        return None
    if not isinstance(value, pd.Series):
        raise TypeError(f"{label} must be a pandas Series")
    normalized = pd.to_numeric(value, errors="coerce")
    if set(map(str, normalized.index)) != set(assets):
        raise ValueError(f"{label} assets must match returns columns")
    normalized.index = normalized.index.map(str)
    normalized = normalized.reindex(assets)
    if normalized.isna().any() or not np.isfinite(normalized.to_numpy(dtype=float)).all():
        raise ValueError(f"{label} must contain finite values for every asset")
    return normalized.astype(float)


def _project_weights_to_bounds(
    weights: pd.Series,
    *,
    min_weight: float,
    max_weight: float | None,
) -> pd.Series:
    """Project positive preference weights into a feasible long-only box simplex.

    Lower bounds are assigned first and the remaining mass is distributed in
    proportion to the original preferences, repeatedly saturating upper bounds.
    This avoids the common `clip -> normalize` bug where normalization can push a
    previously clipped lower-bound weight below the promised minimum again.
    """

    numeric = pd.to_numeric(weights, errors="coerce").astype(float)
    if numeric.isna().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("optimizer preference weights must be finite")
    if (numeric < 0).any():
        raise ValueError("optimizer preference weights must be non-negative")
    total = float(numeric.sum())
    if total <= 0:
        numeric[:] = 1.0
        total = float(numeric.sum())
    preference = numeric / total

    count = len(preference)
    upper = 1.0 if max_weight is None else max_weight
    result = np.full(count, min_weight, dtype=float)
    capacity = np.full(count, upper - min_weight, dtype=float)
    remaining = 1.0 - min_weight * count
    preference_values = preference.to_numpy(dtype=float)
    tolerance = 1e-12

    for _ in range(count + 2):
        if remaining <= tolerance:
            break
        eligible = capacity > tolerance
        if not eligible.any():
            break
        allocation_preference = np.where(eligible, preference_values, 0.0)
        preference_sum = float(allocation_preference.sum())
        if preference_sum <= tolerance:
            allocation_preference = eligible.astype(float)
            preference_sum = float(allocation_preference.sum())
        proposed = remaining * allocation_preference / preference_sum
        allocation = np.minimum(proposed, capacity)
        used = float(allocation.sum())
        result += allocation
        capacity -= allocation
        remaining -= used
        if used <= tolerance:
            break

    if abs(remaining) > 1e-9:
        raise ValueError("failed to project optimizer weights into requested bounds")
    return pd.Series(result, index=preference.index, dtype=float)


@dataclass(frozen=True)
class PortfolioOptimizationRequest:
    """Stable optimizer input independent of any third-party solver object."""

    returns: pd.DataFrame
    expected_returns: pd.Series | None = None
    previous_weights: pd.Series | None = None
    benchmark_weights: pd.Series | None = None
    min_weight: float = 0.0
    max_weight: float | None = None
    covariance_shrinkage: float = 0.0
    long_only: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.returns, pd.DataFrame):
            raise TypeError("returns must be a pandas DataFrame")
        if self.returns.empty or self.returns.shape[1] == 0:
            raise ValueError("returns must contain at least one asset")
        assets = tuple(map(str, self.returns.columns))
        if len(set(assets)) != len(assets):
            raise ValueError("returns columns must be unique")
        numeric = self.returns.apply(pd.to_numeric, errors="coerce")
        if numeric.dropna(how="all").empty:
            raise ValueError("returns must contain numeric observations")
        object.__setattr__(self, "returns", numeric.set_axis(assets, axis=1))
        object.__setattr__(
            self,
            "expected_returns",
            _numeric_series_for_assets(self.expected_returns, assets, label="expected_returns"),
        )
        object.__setattr__(
            self,
            "previous_weights",
            _numeric_series_for_assets(self.previous_weights, assets, label="previous_weights"),
        )
        object.__setattr__(
            self,
            "benchmark_weights",
            _numeric_series_for_assets(self.benchmark_weights, assets, label="benchmark_weights"),
        )
        if self.min_weight < 0:
            raise ValueError("min_weight must be >= 0")
        if self.max_weight is not None and self.max_weight <= 0:
            raise ValueError("max_weight must be > 0")
        if self.max_weight is not None and self.min_weight > self.max_weight:
            raise ValueError("min_weight must be <= max_weight")
        if not 0.0 <= self.covariance_shrinkage <= 1.0:
            raise ValueError("covariance_shrinkage must be in [0, 1]")
        if not self.long_only:
            raise ValueError("native optimizer baselines currently require long_only=True")
        count = len(assets)
        if self.min_weight * count > 1.0 + 1e-12:
            raise ValueError("min_weight is infeasible for the number of assets")
        if self.max_weight is not None and self.max_weight * count < 1.0 - 1e-12:
            raise ValueError("max_weight is infeasible for the number of assets")

    @property
    def assets(self) -> tuple[str, ...]:
        return tuple(map(str, self.returns.columns))


@dataclass(frozen=True)
class PortfolioOptimizationResult:
    backend_name: str
    weights: pd.Series
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = PORTFOLIO_OPTIMIZATION_RESULT_SCHEMA

    def validate(self, request: PortfolioOptimizationRequest) -> None:
        if not self.backend_name.strip():
            raise ValueError("backend_name must be non-empty")
        if self.schema_version != PORTFOLIO_OPTIMIZATION_RESULT_SCHEMA:
            raise ValueError(f"unsupported optimization result schema {self.schema_version!r}")
        if not isinstance(self.weights, pd.Series):
            raise TypeError("weights must be a pandas Series")
        weights = pd.to_numeric(self.weights, errors="coerce")
        weights.index = weights.index.map(str)
        if set(weights.index) != set(request.assets):
            raise ValueError("weights assets must match optimization request")
        weights = weights.reindex(request.assets)
        values = weights.to_numpy(dtype=float)
        if np.isnan(values).any() or not np.isfinite(values).all():
            raise ValueError("weights must be finite")
        if abs(float(weights.sum()) - 1.0) > 1e-8:
            raise ValueError("weights must sum to 1")
        if request.long_only and (weights < -1e-12).any():
            raise ValueError("long-only weights must be non-negative")
        if (weights < request.min_weight - 1e-12).any():
            raise ValueError("weights violate min_weight")
        if request.max_weight is not None and (weights > request.max_weight + 1e-12).any():
            raise ValueError("weights violate max_weight")
        _validate_json_scalars(self.diagnostics)


@runtime_checkable
class PortfolioOptimizerBackend(Protocol):
    name: str

    def run(self, request: PortfolioOptimizationRequest) -> PortfolioOptimizationResult: ...


class OptimizerRegistry:
    """Explicit registry; external optimizer adapters never leak their native objects."""

    def __init__(self) -> None:
        self._backends: dict[str, PortfolioOptimizerBackend] = {}

    def register(self, backend: PortfolioOptimizerBackend) -> None:
        name = str(backend.name).strip()
        if not name:
            raise ValueError("optimizer backend name must be non-empty")
        if name in self._backends:
            raise ValueError(f"optimizer backend already registered: {name}")
        self._backends[name] = backend

    def get(self, name: str) -> PortfolioOptimizerBackend:
        try:
            return self._backends[name]
        except KeyError as exc:
            known = ", ".join(sorted(self._backends)) or "<none>"
            raise KeyError(f"Unknown optimizer backend {name!r}; registered: {known}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._backends))

    def run(self, name: str, request: PortfolioOptimizationRequest) -> PortfolioOptimizationResult:
        result = self.get(name).run(request)
        result.validate(request)
        return result


class EqualWeightOptimizerBackend:
    name = "native.equal_weight"

    def run(self, request: PortfolioOptimizationRequest) -> PortfolioOptimizationResult:
        weight = 1.0 / len(request.assets)
        result = PortfolioOptimizationResult(
            backend_name=self.name,
            weights=pd.Series(weight, index=request.assets, dtype=float),
            diagnostics={"method": "equal_weight", "asset_count": len(request.assets)},
        )
        result.validate(request)
        return result


class HrpOptimizerBackend:
    name = "native.hrp"

    def run(self, request: PortfolioOptimizationRequest) -> PortfolioOptimizationResult:
        if len(request.assets) < 2:
            result = PortfolioOptimizationResult(
                backend_name=self.name,
                weights=pd.Series(1.0, index=request.assets, dtype=float),
                diagnostics={"method": "hrp", "fallback": "single_asset"},
            )
            result.validate(request)
            return result
        hrp = hierarchical_risk_parity(
            request.returns,
            config=HrpConfig(
                shrinkage=request.covariance_shrinkage,
                min_weight=0.0,
                max_weight=None,
            ),
        )
        projected = _project_weights_to_bounds(
            hrp.weights.reindex(request.assets).fillna(0.0),
            min_weight=request.min_weight,
            max_weight=request.max_weight,
        )
        result = PortfolioOptimizationResult(
            backend_name=self.name,
            weights=projected,
            diagnostics={
                "method": "hrp",
                "ordered_assets": list(hrp.ordered_assets),
                "covariance_shrinkage": request.covariance_shrinkage,
                "bounds_projected": request.min_weight > 0 or request.max_weight is not None,
            },
        )
        result.validate(request)
        return result


def _validate_json_scalars(value: Any) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, (float, np.floating)):
        if not isfinite(float(value)):
            raise ValueError("diagnostics must not contain non-finite numbers")
        return
    if isinstance(value, np.integer):
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("diagnostics mapping keys must be strings")
            _validate_json_scalars(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _validate_json_scalars(item)
        return
    raise TypeError(f"unsupported optimizer diagnostics value: {type(value).__name__}")


__all__ = [
    "PORTFOLIO_OPTIMIZATION_RESULT_SCHEMA",
    "EqualWeightOptimizerBackend",
    "HrpOptimizerBackend",
    "OptimizerRegistry",
    "PortfolioOptimizationRequest",
    "PortfolioOptimizationResult",
    "PortfolioOptimizerBackend",
]
