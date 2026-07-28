"""Position post-processing: cash gross overlay application."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from portfolio_backtester._symbol_utils import resolve_data_input_path

from ._postprocess_shared import _as_mapping, _cfg_enabled, _load_table

logger = logging.getLogger("portfolio_backtester")


def _apply_cash_gross_overlay(
    positions: pd.DataFrame,
    *,
    eval_df_full: pd.DataFrame,
    cfg: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not _cfg_enabled(cfg):
        return positions, {"enabled": False}
    work = positions.copy()
    if work.empty:
        return work, {"enabled": True, "period_count": 0}

    diagnostics = _build_overlay_diagnostics(work, eval_df_full=eval_df_full, cfg=cfg)
    target_gross = _resolve_target_gross_by_date(diagnostics, cfg=cfg)
    _validate_target_gross(target_gross, allow_leverage=bool(cfg.get("allow_leverage", False)))

    gross_before = _gross_by_rebalance(work)
    multiplier = (target_gross / gross_before).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    work["weight_before_cash_overlay"] = pd.to_numeric(work["weight"], errors="coerce").fillna(0.0)
    keys = _date_key_series(work["rebalance_date"])
    work["cash_gross_target"] = keys.map(target_gross)
    work["cash_gross_multiplier"] = keys.map(multiplier).fillna(1.0)
    work["weight"] = work["weight_before_cash_overlay"] * work["cash_gross_multiplier"]
    work["cash_weight"] = (1.0 - work["cash_gross_target"]).clip(lower=0.0)
    logger.info("Applied cash gross overlay to %s rebalance dates.", int(target_gross.shape[0]))
    return work, {
        "enabled": True,
        "period_count": int(target_gross.shape[0]),
        "avg_target_gross": float(target_gross.mean()) if not target_gross.empty else np.nan,
        "min_target_gross": float(target_gross.min()) if not target_gross.empty else np.nan,
        "max_target_gross": float(target_gross.max()) if not target_gross.empty else np.nan,
        "schedule_file": str(resolve_data_input_path(str(cfg["schedule_file"])))
        if cfg.get("schedule_file")
        else None,
    }


def _build_overlay_diagnostics(
    positions: pd.DataFrame,
    *,
    eval_df_full: pd.DataFrame,
    cfg: Mapping[str, Any],
) -> pd.DataFrame:
    gross = _gross_by_rebalance(positions)
    counts = positions.groupby(_date_key_series(positions["rebalance_date"]))["symbol"].nunique()
    diagnostics = pd.DataFrame(
        {
            "rebalance_key": gross.index.astype(str),
            "gross_before_overlay": gross.to_numpy(dtype=float),
        }
    )
    diagnostics["position_count"] = diagnostics["rebalance_key"].map(counts).fillna(0).astype(int)
    diagnostics = _merge_overlay_source(diagnostics, cfg=cfg)
    if cfg.get("diagnostics_file") or cfg.get("schedule_file"):
        return diagnostics
    return _merge_eval_date_features(diagnostics, eval_df_full=eval_df_full)


def _merge_overlay_source(
    diagnostics: pd.DataFrame,
    *,
    cfg: Mapping[str, Any],
) -> pd.DataFrame:
    source_file = cfg.get("diagnostics_file") or cfg.get("schedule_file")
    if not source_file:
        return diagnostics
    source = _load_table(str(source_file)).copy()
    date_col = _overlay_date_col(source, cfg=cfg)
    source["rebalance_key"] = _date_key_series(source[date_col])
    return diagnostics.merge(source.drop(columns=[date_col], errors="ignore"), on="rebalance_key")


def _merge_eval_date_features(
    diagnostics: pd.DataFrame,
    *,
    eval_df_full: pd.DataFrame,
) -> pd.DataFrame:
    if eval_df_full.empty or "trade_date" not in eval_df_full.columns:
        return diagnostics
    numeric_cols = [
        col
        for col in eval_df_full.columns
        if col not in {"trade_date", "symbol"} and pd.api.types.is_numeric_dtype(eval_df_full[col])
    ]
    if not numeric_cols:
        return diagnostics
    daily = eval_df_full.copy()
    daily["rebalance_key"] = _date_key_series(daily["trade_date"])
    daily_features = daily.groupby("rebalance_key")[numeric_cols].mean(numeric_only=True)
    return diagnostics.merge(daily_features, on="rebalance_key", how="left")


def _resolve_target_gross_by_date(
    diagnostics: pd.DataFrame,
    *,
    cfg: Mapping[str, Any],
) -> pd.Series:
    schedule_col = str(cfg.get("gross_col") or cfg.get("target_gross_col") or "target_gross")
    if schedule_col in diagnostics.columns:
        return _target_gross_series(diagnostics, schedule_col)
    if cfg.get("gross_multiplier_col") in diagnostics.columns:
        return _target_gross_series(diagnostics, str(cfg["gross_multiplier_col"]))
    if cfg.get("target_gross") is not None:
        values = pd.Series(float(cfg["target_gross"]), index=diagnostics.index, dtype=float)
        return pd.Series(values.to_numpy(), index=diagnostics["rebalance_key"].astype(str))

    targets = []
    default_gross = _default_cash_overlay_gross(cfg)
    for row in diagnostics.to_dict(orient="records"):
        target = _target_gross_for_row(row, cfg=cfg, default_gross=default_gross)
        targets.append(target)
    return pd.Series(targets, index=diagnostics["rebalance_key"].astype(str), dtype=float)


def _target_gross_series(diagnostics: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(diagnostics[column], errors="coerce")
    return pd.Series(values.to_numpy(), index=diagnostics["rebalance_key"].astype(str)).dropna()


def _default_cash_overlay_gross(cfg: Mapping[str, Any]) -> float:
    value = cfg.get(
        "default_gross",
        cfg.get("default_target_gross", cfg.get("default_gross_multiplier", 1.0)),
    )
    return float(value)


def _target_gross_for_row(
    row: Mapping[str, Any],
    *,
    cfg: Mapping[str, Any],
    default_gross: float,
) -> float:
    tiers = cfg.get("tiers")
    if not isinstance(tiers, list):
        return default_gross
    for tier in tiers:
        if not isinstance(tier, Mapping):
            continue
        conditions = _tier_conditions(tier)
        if _conditions_match(row, _as_mapping(conditions)):
            value = tier.get("target_gross", tier.get("gross", tier.get("gross_multiplier")))
            return float(value)
    return default_gross


def _tier_conditions(tier: Mapping[str, Any]) -> Mapping[str, Any]:
    explicit = tier.get("when", tier.get("conditions"))
    if isinstance(explicit, Mapping):
        return explicit
    value_keys = {"target_gross", "gross", "gross_multiplier", "name", "label"}
    return {key: value for key, value in tier.items() if key not in value_keys}


def _conditions_match(row: Mapping[str, Any], conditions: Mapping[str, Any]) -> bool:
    for key, expected in conditions.items():
        if key.startswith("min_"):
            column = key.removeprefix("min_")
            if float(row.get(column, np.nan)) < float(expected):
                return False
        elif key.startswith("max_"):
            column = key.removeprefix("max_")
            if float(row.get(column, np.nan)) > float(expected):
                return False
        elif key.endswith(("_min", "_gte")):
            column = key.rsplit("_", 1)[0]
            if float(row.get(column, np.nan)) < float(expected):
                return False
        elif key.endswith(("_max", "_lte")):
            column = key.rsplit("_", 1)[0]
            if float(row.get(column, np.nan)) > float(expected):
                return False
        elif row.get(key) != expected:
            return False
    return True


def _validate_target_gross(target_gross: pd.Series, *, allow_leverage: bool) -> None:
    if target_gross.empty:
        raise SystemExit("Cash gross overlay produced no target gross schedule.")
    if target_gross.isna().any() or (target_gross < 0).any():
        raise SystemExit("Cash gross overlay target gross values must be finite and non-negative.")
    if not allow_leverage and (target_gross > 1.0 + 1e-12).any():
        raise SystemExit(
            "Cash gross overlay target gross values above 1.0 require allow_leverage=true."
        )


def _gross_by_rebalance(positions: pd.DataFrame) -> pd.Series:
    keys = _date_key_series(positions["rebalance_date"])
    gross = pd.to_numeric(positions["weight"], errors="coerce").abs().groupby(keys).sum()
    return gross.astype(float).replace(0.0, np.nan)


def _overlay_date_col(frame: pd.DataFrame, *, cfg: Mapping[str, Any]) -> str:
    configured = cfg.get("schedule_rebalance_col") or cfg.get("rebalance_col")
    candidates = [configured, "rebalance_date", "trade_date", "date"]
    for candidate in candidates:
        if candidate and str(candidate) in frame.columns:
            return str(candidate)
    raise SystemExit("Cash gross overlay schedule is missing a rebalance_date column.")


def _date_key_series(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    compact = text.str.replace("-", "", regex=False)
    compact_mask = compact.str.fullmatch(r"\d{8}")
    parsed = pd.to_datetime(text, errors="coerce")
    keys = parsed.dt.strftime("%Y%m%d").mask(compact_mask, compact)
    return keys.astype(str)
