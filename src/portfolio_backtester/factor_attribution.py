"""Framework-neutral active return and risk attribution primitives."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
import pandas as pd

FACTOR_RETURN_ATTRIBUTION_SCHEMA = "factor_return_attribution.v1"
FACTOR_RISK_ATTRIBUTION_SCHEMA = "factor_risk_attribution.v1"


def _series(value: pd.Series, *, label: str) -> pd.Series:
    if not isinstance(value, pd.Series):
        raise TypeError(f"{label} must be a pandas Series")
    normalized = pd.to_numeric(value, errors="coerce").astype(float)
    normalized.index = normalized.index.map(str)
    if normalized.index.has_duplicates:
        raise ValueError(f"{label} index must be unique")
    if normalized.isna().any() or not np.isfinite(normalized.to_numpy()).all():
        raise ValueError(f"{label} must contain finite numeric values")
    return normalized


def _frame(value: pd.DataFrame, *, label: str) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"{label} must be a pandas DataFrame")
    normalized = value.apply(pd.to_numeric, errors="coerce").astype(float)
    normalized.index = normalized.index.map(str)
    normalized.columns = normalized.columns.map(str)
    if normalized.index.has_duplicates or normalized.columns.has_duplicates:
        raise ValueError(f"{label} axes must be unique")
    if normalized.isna().any().any() or not np.isfinite(normalized.to_numpy()).all():
        raise ValueError(f"{label} must contain finite numeric values")
    return normalized


def _aligned_inputs(
    weights: pd.Series,
    benchmark_weights: pd.Series,
    exposures: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    portfolio = _series(weights, label="weights")
    benchmark = _series(benchmark_weights, label="benchmark_weights")
    exposure = _frame(exposures, label="exposures")
    assets = set(portfolio.index)
    if set(benchmark.index) != assets or set(exposure.index) != assets:
        raise ValueError("weights, benchmark_weights, and exposures assets must match exactly")
    ordered = tuple(portfolio.index)
    return portfolio.reindex(ordered), benchmark.reindex(ordered), exposure.reindex(ordered)


@dataclass(frozen=True)
class FactorReturnAttribution:
    active_exposures: pd.Series
    factor_contributions: pd.Series
    active_return: float
    specific_return: float
    cost_contribution: float
    schema_version: str = FACTOR_RETURN_ATTRIBUTION_SCHEMA

    @property
    def reconciled_active_return(self) -> float:
        return float(self.factor_contributions.sum() + self.specific_return + self.cost_contribution)

    def receipt(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "active_return": self.active_return,
            "specific_return": self.specific_return,
            "cost_contribution": self.cost_contribution,
            "reconciled_active_return": self.reconciled_active_return,
            "active_exposures": self.active_exposures.to_dict(),
            "factor_contributions": self.factor_contributions.to_dict(),
        }


@dataclass(frozen=True)
class FactorRiskAttribution:
    active_exposures: pd.Series
    factor_variance_contributions: pd.Series
    specific_variance_contribution: float
    active_variance: float
    schema_version: str = FACTOR_RISK_ATTRIBUTION_SCHEMA

    def receipt(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "active_variance": self.active_variance,
            "specific_variance_contribution": self.specific_variance_contribution,
            "active_exposures": self.active_exposures.to_dict(),
            "factor_variance_contributions": self.factor_variance_contributions.to_dict(),
        }


def attribute_factor_return(
    *,
    weights: pd.Series,
    benchmark_weights: pd.Series,
    exposures: pd.DataFrame,
    factor_returns: pd.Series,
    portfolio_return: float,
    benchmark_return: float,
    transaction_cost: float = 0.0,
) -> FactorReturnAttribution:
    """Decompose net active return into factor, specific, and transaction-cost terms."""

    portfolio, benchmark, exposure = _aligned_inputs(weights, benchmark_weights, exposures)
    factors = tuple(exposure.columns)
    factor_return = _series(factor_returns, label="factor_returns")
    if set(factor_return.index) != set(factors):
        raise ValueError("factor_returns factors must match exposure factors")
    factor_return = factor_return.reindex(factors)
    if not isfinite(float(portfolio_return)) or not isfinite(float(benchmark_return)):
        raise ValueError("portfolio_return and benchmark_return must be finite")
    if not isfinite(float(transaction_cost)) or transaction_cost < 0:
        raise ValueError("transaction_cost must be finite and >= 0")

    active_weights = portfolio - benchmark
    active_exposures = exposure.T.dot(active_weights).astype(float)
    factor_contributions = (active_exposures * factor_return).astype(float)
    active_return = float(portfolio_return - benchmark_return)
    cost_contribution = -float(transaction_cost)
    specific_return = float(active_return - factor_contributions.sum() - cost_contribution)
    return FactorReturnAttribution(
        active_exposures=active_exposures,
        factor_contributions=factor_contributions,
        active_return=active_return,
        specific_return=specific_return,
        cost_contribution=cost_contribution,
    )


def attribute_factor_risk(
    *,
    weights: pd.Series,
    benchmark_weights: pd.Series,
    exposures: pd.DataFrame,
    factor_covariance: pd.DataFrame,
    specific_risk: pd.Series,
) -> FactorRiskAttribution:
    """Attribute active variance to factor and independent specific-risk components."""

    portfolio, benchmark, exposure = _aligned_inputs(weights, benchmark_weights, exposures)
    factors = tuple(exposure.columns)
    covariance = _frame(factor_covariance, label="factor_covariance")
    if set(covariance.index) != set(factors) or set(covariance.columns) != set(factors):
        raise ValueError("factor_covariance factors must match exposure factors")
    covariance = covariance.reindex(index=factors, columns=factors)
    if not np.allclose(covariance.to_numpy(), covariance.to_numpy().T, atol=1e-12, rtol=1e-12):
        raise ValueError("factor_covariance must be symmetric")

    specific = _series(specific_risk, label="specific_risk")
    if set(specific.index) != set(portfolio.index):
        raise ValueError("specific_risk assets must match portfolio assets")
    specific = specific.reindex(portfolio.index)
    if (specific < 0).any():
        raise ValueError("specific_risk must be >= 0")

    active_weights = portfolio - benchmark
    active_exposures = exposure.T.dot(active_weights).astype(float)
    marginal_factor_variance = covariance.dot(active_exposures)
    factor_contributions = (active_exposures * marginal_factor_variance).astype(float)
    specific_variance = float(np.square(active_weights * specific).sum())
    active_variance = float(factor_contributions.sum() + specific_variance)
    if active_variance < -1e-12:
        raise ValueError("factor covariance implies negative active variance")
    active_variance = max(active_variance, 0.0)
    return FactorRiskAttribution(
        active_exposures=active_exposures,
        factor_variance_contributions=factor_contributions,
        specific_variance_contribution=specific_variance,
        active_variance=active_variance,
    )


__all__ = [
    "FACTOR_RETURN_ATTRIBUTION_SCHEMA",
    "FACTOR_RISK_ATTRIBUTION_SCHEMA",
    "FactorReturnAttribution",
    "FactorRiskAttribution",
    "attribute_factor_return",
    "attribute_factor_risk",
]
