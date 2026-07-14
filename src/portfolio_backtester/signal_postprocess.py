from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def neutralize_cross_sectional_series(
    data: pd.DataFrame,
    column: str,
    controls: Sequence[str],
    *,
    strength: float = 1.0,
    min_obs: int | None = None,
) -> pd.Series:
    control_cols = [str(col).strip() for col in controls if str(col).strip()]
    if not control_cols:
        return data[column].copy()
    if strength < 0:
        raise ValueError("strength must be >= 0.")
    if strength == 0:
        return data[column].copy()

    out = data[column].copy()
    required_obs = max(
        int(min_obs) if min_obs is not None else 0,
        len(control_cols) + 1,
    )
    if required_obs <= 0:
        required_obs = len(control_cols) + 1

    for _, group in data.groupby("trade_date", sort=False):
        if group.empty:
            continue
        valid = group[column].notna()
        for control_col in control_cols:
            valid &= group[control_col].notna()
        if int(valid.sum()) < required_obs:
            continue

        group_valid = group.loc[valid, [column, *control_cols]].copy()
        y = pd.to_numeric(group_valid[column], errors="coerce").to_numpy(dtype=float)
        x = group_valid[control_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        if y.size < required_obs or x.ndim != 2 or x.shape[0] != y.size:
            continue

        design = np.column_stack([np.ones(y.size, dtype=float), x])
        coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
        fitted_exposure = x @ coeffs[1:]
        out.loc[group_valid.index] = y - strength * fitted_exposure
    return out


def rank_blend_cross_sectional_series(
    data: pd.DataFrame,
    column: str,
    overlays: Sequence[str],
    *,
    strength: float,
) -> pd.Series:
    overlay_cols = [str(col).strip() for col in overlays if str(col).strip()]
    if not overlay_cols:
        raise ValueError("rank_blend requires at least one overlay column.")
    if strength < 0 or strength > 1:
        raise ValueError("strength must be between 0 and 1.")
    required_cols = ["trade_date", column, *overlay_cols]
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        missing_text = ", ".join(sorted(set(missing_cols)))
        raise ValueError(f"Score postprocess columns not found: {missing_text}")

    group_keys = data["trade_date"]
    base_rank = (
        pd.to_numeric(data[column], errors="coerce")
        .groupby(group_keys, sort=False)
        .rank(
            method="average",
            pct=True,
        )
    )
    overlay_values = data[overlay_cols].apply(pd.to_numeric, errors="coerce")
    overlay_ranks = overlay_values.groupby(group_keys, sort=False).rank(
        method="average",
        pct=True,
    )
    overlay_rank = overlay_ranks.mean(axis=1, skipna=True)
    blended = (1.0 - strength) * base_rank + strength * overlay_rank
    return blended.where(overlay_rank.notna(), base_rank)


def apply_score_postprocess(
    data: pd.DataFrame,
    column: str,
    *,
    method: str,
    columns: Sequence[str],
    strength: float = 1.0,
    min_obs: int | None = None,
) -> pd.Series:
    method_text = str(method or "none").strip().lower()
    if method_text == "none":
        return data[column].copy()
    missing_cols = [col for col in columns if col not in data.columns]
    if missing_cols:
        missing_text = ", ".join(sorted(set(missing_cols)))
        raise ValueError(f"Score postprocess columns not found: {missing_text}")
    if method_text == "neutralize":
        return neutralize_cross_sectional_series(
            data,
            column,
            columns,
            strength=strength,
            min_obs=min_obs,
        )
    if method_text == "rank_blend":
        return rank_blend_cross_sectional_series(
            data,
            column,
            columns,
            strength=strength,
        )
    raise ValueError(f"Unsupported score postprocess method: {method}")


def apply_score_postprocess_inplace(
    frame: pd.DataFrame,
    column: str,
    *,
    method: str,
    columns: Sequence[str],
    strength: float,
    min_obs: int | None,
) -> None:
    if method == "none":
        return
    frame[column] = apply_score_postprocess(
        frame,
        column,
        method=method,
        columns=columns,
        strength=strength,
        min_obs=min_obs,
    )


__all__ = [
    "apply_score_postprocess",
    "apply_score_postprocess_inplace",
    "neutralize_cross_sectional_series",
    "rank_blend_cross_sectional_series",
]
