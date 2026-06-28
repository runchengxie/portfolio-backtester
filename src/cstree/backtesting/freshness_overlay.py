from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import pandas as pd


def _enabled(cfg: Mapping[str, Any] | None) -> bool:
    if not cfg:
        return False
    return bool(cfg.get("enabled", False))


def _volume_rank_columns(cfg: Mapping[str, Any]) -> list[str]:
    raw = cfg.get("volume_rank_cols") or cfg.get("volume_cols") or cfg.get("columns")
    if raw is None:
        return ["volume_sma5_ratio", "volume_sma20_ratio", "volume_sma60_ratio"]
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, Sequence):
        return [str(col) for col in raw]
    raise ValueError("freshness_overlay volume columns must be a string or sequence.")


def _rank_pct_by_date(frame: pd.DataFrame, column: str, *, date_col: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    return values.groupby(frame[date_col], sort=False).rank(
        pct=True,
        ascending=True,
        method="average",
        na_option="bottom",
    )


def apply_freshness_overlay(
    frame: pd.DataFrame,
    *,
    score_col: str,
    cfg: Mapping[str, Any] | None,
    date_col: str = "trade_date",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Blend a small freshness rank into an existing per-date score column."""

    if not _enabled(cfg):
        return frame, {"enabled": False}
    if frame.empty:
        return frame, {"enabled": True, "status": "empty"}
    if date_col not in frame.columns:
        raise ValueError(f"freshness_overlay date column not found: {date_col}")

    overlay_cfg = dict(cast(Mapping[str, Any], cfg))
    base_score_col = str(overlay_cfg.get("base_score_col") or score_col)
    if base_score_col not in frame.columns:
        if score_col not in frame.columns:
            raise ValueError(f"freshness_overlay score column not found: {base_score_col}")
        base_score_col = score_col

    volume_cols = _volume_rank_columns(overlay_cfg)
    missing = [col for col in volume_cols if col not in frame.columns]
    if missing:
        raise ValueError(f"freshness_overlay missing volume columns: {', '.join(missing)}")

    lambda_value = float(overlay_cfg.get("lambda", overlay_cfg.get("weight", 0.0)))
    if not 0.0 <= lambda_value <= 1.0:
        raise ValueError("freshness_overlay lambda must be between 0 and 1.")

    out = frame.copy()
    output_col = str(overlay_cfg.get("output_col") or score_col)
    preserve_col = overlay_cfg.get("preserve_base_col")
    if preserve_col is None:
        preserve_col = f"{output_col}_base"
    if preserve_col:
        out[str(preserve_col)] = pd.to_numeric(out[base_score_col], errors="coerce")

    base_rank = _rank_pct_by_date(out, base_score_col, date_col=date_col)
    volume_ranks = [_rank_pct_by_date(out, col, date_col=date_col) for col in volume_cols]
    volume_rank = pd.concat(volume_ranks, axis=1).mean(axis=1)
    out[output_col] = (1.0 - lambda_value) * base_rank + lambda_value * volume_rank
    out[f"{output_col}_freshness_volume_rank"] = volume_rank

    return out, {
        "enabled": True,
        "name": overlay_cfg.get("name") or "volume_only",
        "lambda": lambda_value,
        "base_score_col": base_score_col,
        "output_col": output_col,
        "volume_rank_cols": volume_cols,
        "rows": int(len(out)),
        "dates": int(pd.Series(out[date_col]).nunique()),
    }


__all__ = ["apply_freshness_overlay"]
