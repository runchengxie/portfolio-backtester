"""Portfolio sizing primitives for calibrated signals and active bets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from math import erf, sqrt
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SizingConfig:
    method: Literal[
        "probability",
        "probability_vol_target",
        "signal_vol_target",
        "confidence_budget",
        "risk_budget",
    ] = "probability_vol_target"
    gross_target: float = 1.0
    volatility_target: float | None = None
    single_name_cap: float | None = None
    step_size: float | None = None
    min_trade_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.gross_target <= 0:
            raise ValueError("gross_target must be > 0")
        if self.volatility_target is not None and self.volatility_target <= 0:
            raise ValueError("volatility_target must be > 0")
        if self.single_name_cap is not None and self.single_name_cap <= 0:
            raise ValueError("single_name_cap must be > 0")
        if self.step_size is not None and not 0 < self.step_size <= 1:
            raise ValueError("step_size must be in (0, 1]")
        if self.min_trade_weight < 0:
            raise ValueError("min_trade_weight must be >= 0")


def probability_to_size(
    probability: pd.Series,
    *,
    classes: int = 2,
    side: pd.Series | float = 1.0,
) -> pd.Series:
    """Translate calibrated probabilities to signed position intensities."""

    if classes < 2:
        raise ValueError("classes must be >= 2")
    values = pd.to_numeric(probability, errors="coerce").clip(0.0, 1.0)
    denominator = np.sqrt(values * (1.0 - values))
    z = (values - 1.0 / classes).div(denominator.replace(0.0, np.nan))
    size = (
        2.0 * z.map(lambda value: _normal_cdf(float(value)) if np.isfinite(value) else np.nan) - 1.0
    )
    side_values = (
        pd.to_numeric(side, errors="coerce").reindex(values.index)
        if isinstance(side, pd.Series)
        else pd.Series(float(side), index=values.index)
    )
    return (size * np.sign(side_values)).clip(-1.0, 1.0).rename("raw_size")


def average_active_bets(
    events: pd.DataFrame,
    *,
    time_index: pd.DatetimeIndex | None = None,
    start_col: str = "label_start",
    end_col: str = "label_end",
    size_col: str = "bet_size",
) -> pd.Series:
    """Average all bet sizes that remain active at each point in time."""

    missing = [column for column in (start_col, end_col, size_col) if column not in events.columns]
    if missing:
        raise ValueError(f"events is missing required columns: {', '.join(missing)}")
    data = events.copy()
    data[start_col] = pd.to_datetime(data[start_col], errors="coerce")
    data[end_col] = pd.to_datetime(data[end_col], errors="coerce")
    data[size_col] = pd.to_numeric(data[size_col], errors="coerce")
    data = data.dropna(subset=[start_col, end_col, size_col])
    if bool((data[end_col] < data[start_col]).any()):
        raise ValueError("event end must be on or after event start")
    if time_index is None:
        times = pd.DatetimeIndex(
            sorted(set(data[start_col].tolist()) | set(data[end_col].tolist()))
        )
    else:
        times = pd.DatetimeIndex(time_index).drop_duplicates().sort_values()
    values: list[float] = []
    for timestamp in times:
        active = data.loc[(data[start_col] <= timestamp) & (data[end_col] >= timestamp), size_col]
        values.append(float(active.mean()) if not active.empty else 0.0)
    return pd.Series(values, index=times, dtype=float, name="active_bet_size")


def discretize_weights(weights: pd.Series, *, step_size: float) -> pd.Series:
    """Discretize weights to reduce small, noisy portfolio changes."""

    if not 0 < step_size <= 1:
        raise ValueError("step_size must be in (0, 1]")
    values = pd.to_numeric(weights, errors="coerce")
    return ((values / step_size).round() * step_size).rename(weights.name)


def build_sized_weights(
    frame: pd.DataFrame,
    *,
    score_col: str,
    config: SizingConfig | None = None,
    probability_col: str = "calibrated_probability",
    volatility_col: str = "predicted_volatility",
    confidence_col: str = "calibrated_confidence",
    risk_budget_col: str = "risk_budget",
    side_col: str | None = None,
) -> pd.Series:
    """Build normalized portfolio weights from calibrated research outputs."""

    cfg = config or SizingConfig()
    if score_col not in frame.columns:
        raise ValueError(f"score column not found: {score_col}")
    side = (
        pd.to_numeric(frame[side_col], errors="coerce")
        if side_col is not None and side_col in frame.columns
        else pd.Series(1.0, index=frame.index)
    )
    if cfg.method.startswith("probability"):
        if probability_col not in frame.columns:
            raise ValueError(f"probability column not found: {probability_col}")
        raw = probability_to_size(frame[probability_col], side=side).abs()
    elif cfg.method == "signal_vol_target":
        raw = pd.to_numeric(frame[score_col], errors="coerce").abs()
    elif cfg.method == "confidence_budget":
        if confidence_col not in frame.columns:
            raise ValueError(f"confidence column not found: {confidence_col}")
        raw = pd.to_numeric(frame[confidence_col], errors="coerce").clip(lower=0.0)
    elif cfg.method == "risk_budget":
        if risk_budget_col not in frame.columns:
            raise ValueError(f"risk budget column not found: {risk_budget_col}")
        raw = pd.to_numeric(frame[risk_budget_col], errors="coerce").clip(lower=0.0)
    else:
        raise ValueError(f"Unsupported sizing method: {cfg.method}")

    if cfg.method in {"probability_vol_target", "signal_vol_target"}:
        if volatility_col not in frame.columns:
            raise ValueError(f"volatility column not found: {volatility_col}")
        volatility = pd.to_numeric(frame[volatility_col], errors="coerce").replace(0.0, np.nan)
        raw = raw.div(volatility.abs())

    raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    weights = _normalize_gross(raw, cfg.gross_target)
    if cfg.single_name_cap is not None:
        weights = _cap_and_redistribute(weights, cfg.single_name_cap, cfg.gross_target)
    if cfg.step_size is not None:
        weights = discretize_weights(weights, step_size=cfg.step_size)
        weights = _normalize_gross(weights.clip(lower=0.0), cfg.gross_target)
    if cfg.min_trade_weight > 0:
        weights = weights.where(weights >= cfg.min_trade_weight, 0.0)
        weights = _normalize_gross(weights, cfg.gross_target)
    return weights.rename("target_weight")


def build_sizing_receipt(
    weights: pd.Series,
    *,
    config: SizingConfig,
    calibration_artifact: str | None = None,
    covariance_artifact: str | None = None,
) -> dict[str, object]:
    """Create a deterministic sizing receipt for lineage sidecars."""

    payload = pd.to_numeric(weights, errors="coerce").fillna(0.0).to_csv(header=False)
    return {
        "schema_version": 1,
        "method": config.method,
        "config": asdict(config),
        "calibration_artifact": calibration_artifact,
        "covariance_artifact": covariance_artifact,
        "target_count": int((weights.abs() > 0).sum()),
        "gross_exposure": float(weights.abs().sum()),
        "maximum_weight": float(weights.abs().max()) if not weights.empty else float("nan"),
        "weights_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }


def write_sizing_receipt(receipt: dict[str, object], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, default=str) + "\n")


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def _normalize_gross(values: pd.Series, gross_target: float) -> pd.Series:
    cleaned = pd.to_numeric(values, errors="coerce").fillna(0.0).clip(lower=0.0)
    total = float(cleaned.sum())
    if total <= 0:
        return pd.Series(0.0, index=cleaned.index, dtype=float)
    return cleaned / total * gross_target


def _cap_and_redistribute(
    values: pd.Series,
    cap: float,
    gross_target: float,
    *,
    max_iter: int = 100,
) -> pd.Series:
    if cap * len(values) + 1e-12 < gross_target:
        raise ValueError("single_name_cap is infeasible for the number of positions")
    weights = values.copy()
    fixed = pd.Series(False, index=weights.index)
    for _ in range(max_iter):
        over = (weights > cap + 1e-12) & ~fixed
        if not bool(over.any()):
            break
        fixed |= over
        weights.loc[fixed] = cap
        residual = gross_target - float(weights.loc[fixed].sum())
        free = ~fixed
        if residual <= 0 or not bool(free.any()):
            break
        base = values.loc[free]
        weights.loc[free] = (
            residual * base / float(base.sum()) if base.sum() > 0 else residual / int(free.sum())
        )
    return weights


__all__ = [
    "SizingConfig",
    "average_active_bets",
    "build_sized_weights",
    "build_sizing_receipt",
    "discretize_weights",
    "probability_to_size",
    "write_sizing_receipt",
]
