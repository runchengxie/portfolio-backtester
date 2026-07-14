"""Hierarchical Risk Parity allocations for assets, sleeves, or model signals."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import pairwise

import numpy as np
import pandas as pd

try:
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform
except Exception:  # pragma: no cover - optional dependency guard
    leaves_list = None
    linkage = None
    squareform = None


@dataclass(frozen=True)
class HrpConfig:
    linkage_method: str = "single"
    shrinkage: float = 0.0
    min_weight: float = 0.0
    max_weight: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.shrinkage <= 1.0:
            raise ValueError("shrinkage must be in [0, 1]")
        if self.min_weight < 0:
            raise ValueError("min_weight must be >= 0")
        if self.max_weight is not None and self.max_weight <= 0:
            raise ValueError("max_weight must be > 0")


@dataclass(frozen=True)
class HrpResult:
    weights: pd.Series
    ordered_assets: tuple[str, ...]
    covariance: pd.DataFrame
    correlation: pd.DataFrame
    config: HrpConfig

    def receipt(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "method": "hrp",
            "ordered_assets": list(self.ordered_assets),
            "config": asdict(self.config),
            "gross_weight": float(self.weights.abs().sum()),
            "maximum_weight": float(self.weights.max()) if not self.weights.empty else float("nan"),
            "minimum_weight": float(self.weights.min()) if not self.weights.empty else float("nan"),
        }


def hierarchical_risk_parity(
    returns: pd.DataFrame,
    *,
    config: HrpConfig | None = None,
) -> HrpResult:
    """Compute HRP weights without inverting the covariance matrix."""

    cfg = config or HrpConfig()
    if linkage is None or leaves_list is None or squareform is None:
        raise RuntimeError("scipy is required for hierarchical risk parity")
    data = returns.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    data = data.dropna(axis=1, how="all")
    if data.shape[1] < 2:
        raise ValueError("HRP requires at least two return series")
    covariance = data.cov(min_periods=2)
    if covariance.isna().any().any():
        raise ValueError("HRP covariance contains missing values")
    covariance = _shrink_covariance(covariance, cfg.shrinkage)
    correlation = _covariance_to_correlation(covariance)
    distance = np.sqrt(np.clip((1.0 - correlation.to_numpy(dtype=float)) / 2.0, 0.0, 1.0))
    condensed = squareform(distance, checks=False)
    tree = linkage(condensed, method=cfg.linkage_method)
    order = leaves_list(tree).astype(int).tolist()
    ordered_assets = tuple(str(covariance.columns[index]) for index in order)
    weights = _recursive_bisection(covariance, list(ordered_assets))
    weights = _apply_bounds(weights, cfg)
    return HrpResult(
        weights=weights,
        ordered_assets=ordered_assets,
        covariance=covariance,
        correlation=correlation,
        config=cfg,
    )


def rolling_hrp_weights(
    returns: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    *,
    lookback: int = 252,
    min_observations: int = 60,
    config: HrpConfig | None = None,
) -> pd.DataFrame:
    """Estimate HRP weights using data strictly before each rebalance date."""

    if lookback <= 0 or min_observations <= 1:
        raise ValueError("lookback and min_observations must be positive")
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise ValueError("returns must use a DatetimeIndex")
    rows: list[pd.Series] = []
    for date in pd.DatetimeIndex(rebalance_dates).sort_values():
        history = returns.loc[returns.index < date].tail(lookback)
        if len(history.dropna(how="all")) < min_observations:
            continue
        result = hierarchical_risk_parity(history, config=config)
        rows.append(result.weights.rename(date))
    if not rows:
        return pd.DataFrame(columns=returns.columns, dtype=float)
    output = pd.DataFrame(rows).fillna(0.0)
    output.index.name = "rebalance_date"
    return output.reindex(columns=returns.columns, fill_value=0.0)


def cluster_stability(
    previous_order: tuple[str, ...] | list[str],
    current_order: tuple[str, ...] | list[str],
) -> float:
    """Measure adjacent-pair stability between two quasi-diagonal orders."""

    previous_pairs = {frozenset(pair) for pair in pairwise(previous_order)}
    current_pairs = {frozenset(pair) for pair in pairwise(current_order)}
    union = previous_pairs | current_pairs
    return len(previous_pairs & current_pairs) / len(union) if union else 1.0


def _shrink_covariance(covariance: pd.DataFrame, shrinkage: float) -> pd.DataFrame:
    if shrinkage <= 0:
        return covariance
    diagonal = pd.DataFrame(
        np.diag(np.diag(covariance.to_numpy(dtype=float))),
        index=covariance.index,
        columns=covariance.columns,
    )
    return covariance * (1.0 - shrinkage) + diagonal * shrinkage


def _covariance_to_correlation(covariance: pd.DataFrame) -> pd.DataFrame:
    scale = np.sqrt(np.diag(covariance.to_numpy(dtype=float)))
    denominator = np.outer(scale, scale)
    correlation = covariance.to_numpy(dtype=float) / denominator
    correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(correlation, 1.0)
    return pd.DataFrame(correlation, index=covariance.index, columns=covariance.columns)


def _cluster_variance(covariance: pd.DataFrame, assets: list[str]) -> float:
    sub_covariance = covariance.loc[assets, assets]
    diagonal = np.diag(sub_covariance.to_numpy(dtype=float))
    inverse = np.divide(1.0, diagonal, out=np.zeros_like(diagonal), where=diagonal > 0)
    if inverse.sum() <= 0:
        inverse = np.ones_like(inverse)
    weights = inverse / inverse.sum()
    return float(weights @ sub_covariance.to_numpy(dtype=float) @ weights)


def _recursive_bisection(covariance: pd.DataFrame, ordered_assets: list[str]) -> pd.Series:
    weights = pd.Series(1.0, index=ordered_assets, dtype=float)
    clusters: list[list[str]] = [ordered_assets]
    while clusters:
        next_clusters: list[list[str]] = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            split = len(cluster) // 2
            left = cluster[:split]
            right = cluster[split:]
            left_variance = _cluster_variance(covariance, left)
            right_variance = _cluster_variance(covariance, right)
            total = left_variance + right_variance
            alpha = 0.5 if total <= 0 else 1.0 - left_variance / total
            weights.loc[left] *= alpha
            weights.loc[right] *= 1.0 - alpha
            next_clusters.extend([left, right])
        clusters = next_clusters
    return weights / weights.sum()


def _apply_bounds(weights: pd.Series, config: HrpConfig) -> pd.Series:
    bounded = weights.clip(lower=config.min_weight)
    if config.max_weight is not None:
        if config.max_weight * len(weights) + 1e-12 < 1.0:
            raise ValueError("max_weight is infeasible for the number of assets")
        for _ in range(100):
            over = bounded > config.max_weight + 1e-12
            if not bool(over.any()):
                break
            bounded.loc[over] = config.max_weight
            free = ~over
            residual = 1.0 - float(bounded.loc[over].sum())
            if residual <= 0 or not bool(free.any()):
                break
            base = weights.loc[free]
            bounded.loc[free] = residual * base / float(base.sum())
    return bounded / bounded.sum()


__all__ = [
    "HrpConfig",
    "HrpResult",
    "cluster_stability",
    "hierarchical_risk_parity",
    "rolling_hrp_weights",
]
