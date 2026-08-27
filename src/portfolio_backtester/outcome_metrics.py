"""Distributional and path diagnostics for realized investment outcomes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass(frozen=True, slots=True)
class OutcomeDistributionReport:
    """Immutable summary of realized return and path outcomes."""

    observations: int
    mean_return: float
    median_return: float
    loss_probability: float
    q05_return: float
    q25_return: float
    q75_return: float
    q95_return: float
    cvar_05_return: float
    mean_mfe: float
    median_mfe: float
    mean_mae: float
    median_mae: float
    mean_peak_giveback: float
    p90_peak_giveback: float
    mean_holding_period: float
    median_holding_period: float
    p90_holding_period: float
    schema_version: str = field(default="outcome_distribution.v1", init=False)

    def to_dict(self) -> dict[str, float | int | str]:
        """Return a deterministic JSON-compatible representation."""

        return asdict(self)


def _finite_array(values: Sequence[float], *, label: str) -> np.ndarray:
    array = np.asarray(tuple(values), dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{label} must be one-dimensional")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must contain only finite values")
    return array


def _validate_path_semantics(
    *,
    mfe: np.ndarray,
    mae: np.ndarray,
    peak_giveback: np.ndarray,
    holding_periods: np.ndarray,
) -> None:
    if np.any(mfe < 0):
        raise ValueError("mfe values must be non-negative")
    if np.any(mae > 0):
        raise ValueError("mae values must be non-positive")
    if np.any(peak_giveback < 0):
        raise ValueError("peak_giveback values must be non-negative")
    if np.any(holding_periods < 0):
        raise ValueError("holding_periods values must be non-negative")


def summarize_outcome_distribution(
    *,
    returns: Sequence[float],
    mfe: Sequence[float],
    mae: Sequence[float],
    peak_giveback: Sequence[float],
    holding_periods: Sequence[float],
) -> OutcomeDistributionReport:
    """Summarize realized outcome distribution and path diagnostics.

    MFE and MAE use signed return excursions from entry. Peak giveback is a
    non-negative return-space distance from the best observed mark to exit.
    Holding periods are non-negative caller-defined periods.
    """

    arrays = {
        "returns": _finite_array(returns, label="returns"),
        "mfe": _finite_array(mfe, label="mfe"),
        "mae": _finite_array(mae, label="mae"),
        "peak_giveback": _finite_array(peak_giveback, label="peak_giveback"),
        "holding_periods": _finite_array(holding_periods, label="holding_periods"),
    }
    lengths = {len(values) for values in arrays.values()}
    if lengths == {0}:
        raise ValueError("outcome inputs must not be empty")
    if len(lengths) != 1:
        raise ValueError("outcome inputs must have the same length")

    _validate_path_semantics(
        mfe=arrays["mfe"],
        mae=arrays["mae"],
        peak_giveback=arrays["peak_giveback"],
        holding_periods=arrays["holding_periods"],
    )

    realized_returns = arrays["returns"]
    q05, q25, q75, q95 = np.quantile(realized_returns, [0.05, 0.25, 0.75, 0.95])
    cvar_tail = realized_returns[realized_returns <= q05]

    return OutcomeDistributionReport(
        observations=len(realized_returns),
        mean_return=float(np.mean(realized_returns)),
        median_return=float(np.median(realized_returns)),
        loss_probability=float(np.mean(realized_returns < 0)),
        q05_return=float(q05),
        q25_return=float(q25),
        q75_return=float(q75),
        q95_return=float(q95),
        cvar_05_return=float(np.mean(cvar_tail)),
        mean_mfe=float(np.mean(arrays["mfe"])),
        median_mfe=float(np.median(arrays["mfe"])),
        mean_mae=float(np.mean(arrays["mae"])),
        median_mae=float(np.median(arrays["mae"])),
        mean_peak_giveback=float(np.mean(arrays["peak_giveback"])),
        p90_peak_giveback=float(np.quantile(arrays["peak_giveback"], 0.90)),
        mean_holding_period=float(np.mean(arrays["holding_periods"])),
        median_holding_period=float(np.median(arrays["holding_periods"])),
        p90_holding_period=float(np.quantile(arrays["holding_periods"], 0.90)),
    )


__all__ = ["OutcomeDistributionReport", "summarize_outcome_distribution"]
