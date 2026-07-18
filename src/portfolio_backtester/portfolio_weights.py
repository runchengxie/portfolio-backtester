from __future__ import annotations

import numpy as np
import pandas as pd

from .bet_sizing import SizingConfig, build_sized_weights

__all__ = [
    "build_position_weights",
    "limit_weight_turnover",
    "normalize_position_weights",
    "normalize_weighting_mode",
    "validate_positive_name_invariant",
]

_WEIGHTING_MODES = {
    "equal",
    "signal",
    "sqrt_liquidity",
    "probability",
    "probability_vol_target",
    "signal_vol_target",
    "confidence_budget",
    "risk_budget",
}


def normalize_weighting_mode(weighting: str | None) -> str:
    mode = str(weighting or "equal").strip().lower()
    if mode not in _WEIGHTING_MODES:
        raise ValueError("weighting must be one of: " + ", ".join(sorted(_WEIGHTING_MODES)) + ".")
    return mode


def _equal_weights(holdings: list[str]) -> pd.Series:
    if not holdings:
        return pd.Series(dtype=float)
    return pd.Series(np.repeat(1.0 / len(holdings), len(holdings)), index=holdings, dtype=float)


def normalize_position_weights(weights: pd.Series) -> pd.Series:
    if weights is None or weights.empty:
        return pd.Series(dtype=float)
    cleaned = pd.to_numeric(weights, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if cleaned.empty:
        return pd.Series(dtype=float)
    total = float(cleaned.sum())
    if not np.isfinite(total) or total <= 0:
        return _equal_weights(list(cleaned.index))
    return cleaned / total


def _cap_and_redistribute_positive_weights(
    base: pd.Series,
    caps: pd.Series,
    *,
    max_iter: int = 50,
) -> pd.Series:
    base = pd.to_numeric(base, errors="coerce").fillna(0).clip(lower=0)
    caps = pd.to_numeric(caps, errors="coerce").fillna(0).clip(lower=0)
    if base.empty:
        return pd.Series(dtype=float)
    if base.sum() <= 0:
        base = pd.Series(1.0, index=base.index, dtype=float)
    weights = base / base.sum()
    if caps.sum() <= 0:
        return weights
    if caps.sum() < 1.0:
        return caps / caps.sum()

    fixed = pd.Series(False, index=weights.index)
    for _ in range(max_iter):
        over = (weights > caps + 1e-12) & (~fixed)
        if not over.any():
            break
        fixed |= over
        weights.loc[fixed] = caps.loc[fixed]
        residual = 1.0 - float(weights.loc[fixed].sum())
        free = ~fixed
        if residual <= 1e-12 or not free.any():
            break
        base_free = base.loc[free]
        if base_free.sum() <= 0:
            weights.loc[free] = residual / int(free.sum())
        else:
            weights.loc[free] = residual * base_free / base_free.sum()
    return normalize_position_weights(weights)


def _sqrt_liquidity_weights(
    day: pd.DataFrame, holdings: list[str], liquidity_col: str
) -> pd.Series:
    if not holdings:
        return pd.Series(dtype=float)
    if liquidity_col not in day.columns:
        return _equal_weights(holdings)
    liquidity = pd.to_numeric(
        day.set_index("symbol").reindex(holdings)[liquidity_col],
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan)
    if liquidity.notna().any():
        liquidity = liquidity.fillna(float(liquidity.median()))
    else:
        liquidity = pd.Series(1.0, index=liquidity.index, dtype=float)
    liquidity = liquidity.clip(lower=1.0)
    raw = np.sqrt(liquidity)
    if raw.sum() <= 0:
        return _equal_weights(holdings)
    concentration_cap = min(0.05, 2.5 / len(holdings))
    caps = pd.Series(concentration_cap, index=raw.index, dtype=float)
    return _cap_and_redistribute_positive_weights(raw, caps)


def _calibrated_weights(
    day: pd.DataFrame,
    holdings: list[str],
    pred_col: str,
    *,
    mode: str,
) -> pd.Series:
    selected = day.set_index("symbol").reindex(holdings).copy()
    if selected.empty:
        return pd.Series(dtype=float)
    sized = build_sized_weights(
        selected,
        score_col=pred_col,
        config=SizingConfig(method=mode),  # type: ignore[arg-type]
    )
    return normalize_position_weights(sized.reindex(holdings).fillna(0.0))


def build_position_weights(
    day: pd.DataFrame,
    holdings: list[str],
    pred_col: str,
    *,
    side: str,
    weighting: str = "equal",
    liquidity_col: str = "medadv20_amount",
) -> pd.Series:
    mode = normalize_weighting_mode(weighting)
    base = _equal_weights(holdings)
    if mode == "equal" or base.empty:
        return base

    if mode == "sqrt_liquidity":
        return _sqrt_liquidity_weights(day, holdings, liquidity_col)

    if side not in {"long", "short"}:
        raise ValueError("side must be one of: long, short.")

    if mode in {
        "probability",
        "probability_vol_target",
        "signal_vol_target",
        "confidence_budget",
        "risk_budget",
    }:
        return _calibrated_weights(day, holdings, pred_col, mode=mode)

    signal = pd.to_numeric(
        day.set_index("symbol").reindex(holdings)[pred_col],
        errors="coerce",
    )
    if side == "short":
        signal = -signal
    if signal.empty or signal.isna().all():
        return base

    signal = signal.fillna(float(signal.mean()) if signal.notna().any() else 0.0)
    std = float(signal.std(ddof=0))
    if not np.isfinite(std) or std <= 0:
        return base

    scaled = ((signal - float(signal.mean())) / std).clip(-5.0, 5.0)
    raw = np.exp(scaled.to_numpy(dtype=float))
    total = float(np.sum(raw))
    if not np.isfinite(total) or total <= 0:
        return base
    return normalize_position_weights(pd.Series(raw, index=signal.index, dtype=float))


def limit_weight_turnover(
    previous: pd.Series | None,
    target: pd.Series,
    max_turnover: float | None,
) -> pd.Series:
    if previous is None or previous.empty or target.empty or max_turnover is None:
        return target
    cap = float(max_turnover)
    if cap <= 0:
        return normalize_position_weights(previous)
    symbols = previous.index.union(target.index)
    prev = previous.reindex(symbols).fillna(0.0).astype(float)
    desired = target.reindex(symbols).fillna(0.0).astype(float)
    turnover = float((desired - prev).abs().sum())
    if turnover <= cap or turnover <= 0:
        limited = desired
    else:
        limited = prev + (desired - prev) * (cap / turnover)
    limited = limited[limited.abs() > 1e-12]
    return normalize_position_weights(limited)


def validate_positive_name_invariant(
    weights: pd.Series,
    max_positive_names: int | None,
) -> pd.Series:
    """Fail instead of silently emitting a turnover-cap long tail."""

    if max_positive_names is None:
        return weights
    positive_names = int((pd.to_numeric(weights, errors="coerce").fillna(0.0) > 1e-12).sum())
    if positive_names > max_positive_names:
        raise ValueError(
            "Final portfolio exceeds max_positive_names: "
            f"{positive_names} > {max_positive_names}. Avoid weight interpolation or use an "
            "explicit discrete replacement policy."
        )
    return weights
