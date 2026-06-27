from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from market_data_platform.artifacts import resolve_data_input_path

from .exposure import compute_backtest_exposure_analysis
from .position_backtest import PositionBacktestConfig, run_position_backtest
from .post_buffer_exposure_repair import (
    PostBufferExposureRepairConfig,
    repair_post_buffer_exposure,
)

logger = logging.getLogger("cstree")


def positions_postprocess_enabled(context: Mapping[str, Any]) -> bool:
    return _cfg_enabled(context.get("post_buffer_exposure_repair")) or _cfg_enabled(
        context.get("cash_gross_overlay")
    )


def apply_position_postprocess(
    positions: pd.DataFrame | None,
    *,
    eval_df_full: pd.DataFrame,
    context: Mapping[str, Any],
) -> tuple[pd.DataFrame | None, dict[str, Any], dict[str, pd.DataFrame]]:
    metadata: dict[str, Any] = {
        "schema": "pipeline_position_postprocess.v1",
        "enabled": positions_postprocess_enabled(context),
        "post_buffer_exposure_repair": {"enabled": False},
        "cash_gross_overlay": {"enabled": False},
    }
    artifacts: dict[str, pd.DataFrame] = {}
    if positions is None or positions.empty or not metadata["enabled"]:
        return positions, metadata, artifacts

    repaired, repair_meta, repair_artifacts = _apply_post_buffer_exposure_repair(
        positions,
        eval_df_full=eval_df_full,
        context=context,
        cfg=_as_mapping(context.get("post_buffer_exposure_repair")),
    )
    artifacts.update(repair_artifacts)
    metadata["post_buffer_exposure_repair"] = repair_meta
    overlaid, overlay_meta = _apply_cash_gross_overlay(
        repaired,
        eval_df_full=eval_df_full,
        cfg=_as_mapping(context.get("cash_gross_overlay")),
    )
    metadata["cash_gross_overlay"] = overlay_meta
    return overlaid, metadata, artifacts


def rebuild_backtest_from_positions(
    positions: pd.DataFrame | None,
    bt_result: tuple | None,
    *,
    context: Mapping[str, Any],
) -> tuple | None:
    if not positions_postprocess_enabled(context) or positions is None or positions.empty:
        return bt_result
    if bt_result is None:
        return None
    if not bool(context.get("backtest_long_only", True)):
        raise SystemExit("Position postprocess backtest currently requires long-only positions.")

    _, _, _, _, period_info = bt_result
    if not period_info:
        return bt_result
    execution_model = context["execution_model"]
    entry_price_col = execution_model.entry_policy.price_col
    exit_price_col = execution_model.exit_policy.price_col
    if entry_price_col != exit_price_col:
        raise SystemExit(
            "Position postprocess backtest requires the same entry and exit price column; "
            f"got entry={entry_price_col}, exit={exit_price_col}."
        )

    preserve_gross = bool(
        context.get("backtest_preserve_gross_exposure")
        or _cfg_enabled(context.get("cash_gross_overlay"))
    )
    config = PositionBacktestConfig(
        price_col=entry_price_col,
        transaction_cost_bps=float(context.get("backtest_cost_bps_effective", 0.0)),
        trading_days_per_year=int(context.get("backtest_trading_days_per_year", 252)),
        long_only=True,
        preserve_gross_exposure=preserve_gross,
        exit_price_policy=context.get("backtest_exit_price_policy", "strict"),
        exit_fallback_policy=context.get("backtest_exit_fallback_policy", "ffill"),
        tradable_col=context.get("backtest_tradable_col"),
    )
    try:
        result = run_position_backtest(
            positions=positions,
            pricing=context["backtest_pricing_df"],
            periods=pd.DataFrame(period_info),
            config=config,
        )
    except ValueError as exc:
        raise SystemExit(f"Position postprocess backtest failed: {exc}") from exc

    net_series = _series_from_position_backtest(result.net_returns, "net_return")
    gross_series = _series_from_position_backtest(result.gross_returns, "gross_return")
    periods = result.periods.to_dict(orient="records")
    turnover_series = pd.Series(
        pd.to_numeric(result.periods["turnover"], errors="coerce").to_numpy(dtype=float),
        index=pd.to_datetime(result.periods["exit_date"], errors="coerce"),
        name="turnover",
    )
    stats = dict(result.summary.get("stats", {}))
    stats["position_postprocess"] = True
    return stats, net_series, gross_series, turnover_series, periods


def _apply_post_buffer_exposure_repair(
    positions: pd.DataFrame,
    *,
    eval_df_full: pd.DataFrame,
    context: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, pd.DataFrame]]:
    if not _cfg_enabled(cfg):
        return positions, {"enabled": False}, {}
    source_file = _first_path_value(cfg, "source_file", "source_path", "exposure_source_file")
    source = _load_table(source_file) if source_file is not None else eval_df_full.copy()
    exposure = _compute_repair_exposure(positions, source=source, context=context)
    artifacts = _repair_exposure_artifacts(exposure)

    breaches_file = _first_path_value(cfg, "breaches_file", "breach_file", "breaches_path")
    if breaches_file is not None:
        breaches = _load_table(breaches_file)
        breach_source = "file"
    else:
        breaches = _auto_repair_breaches(exposure, cfg=cfg)
        breach_source = "auto_exposure"
    artifacts["breaches"] = breaches

    repair_cfg = _repair_config_from_mapping(cfg)
    result = repair_post_buffer_exposure(
        positions,
        source,
        breaches,
        config=repair_cfg,
    )
    logger.info("Applied post-buffer exposure repair: %s actions.", len(result.actions))
    return (
        result.positions,
        {
            "enabled": True,
            "breach_source": breach_source,
            "breach_count": int(breaches.shape[0]),
            "actions": result.actions,
            "action_count": len(result.actions),
            "breaches_file": str(resolve_data_input_path(breaches_file)) if breaches_file else None,
            "source_file": str(resolve_data_input_path(source_file)) if source_file else None,
            "pre_repair_exposure": _exposure_metadata(exposure),
        },
        artifacts,
    )


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


def _compute_repair_exposure(
    positions: pd.DataFrame,
    *,
    source: pd.DataFrame,
    context: Mapping[str, Any],
) -> dict[str, Any]:
    exposure_source = context.get("exposure_source_df")
    scored_data = exposure_source if isinstance(exposure_source, pd.DataFrame) else source
    return compute_backtest_exposure_analysis(
        scored_data,
        positions,
        pricing_data=context.get("backtest_pricing_df"),
        price_col=str(context.get("price_col", "close")),
        benchmark_df=context.get("benchmark_df"),
        benchmark_return_series=context.get("benchmark_return_series"),
        market_cap_col=context.get("fundamentals_mcap_col"),
        industry_columns=context.get("industry_columns", []),
        industry_source_data=context.get("industry_source_df"),
    )


def _repair_exposure_artifacts(exposure: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    artifacts: dict[str, pd.DataFrame] = {}
    for source_key, artifact_key in (
        ("style", "pre_repair_style"),
        ("industry", "pre_repair_industry"),
        ("active_summary", "pre_repair_active_summary"),
    ):
        frame = exposure.get(source_key)
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            artifacts[artifact_key] = frame
    return artifacts


def _exposure_metadata(exposure: Mapping[str, Any]) -> dict[str, Any]:
    style_summary = exposure.get("style_summary")
    industry_summary = exposure.get("industry_summary")
    style_meta = style_summary if isinstance(style_summary, Mapping) else {}
    industry_meta = industry_summary if isinstance(industry_summary, Mapping) else {}
    return {
        "latest_rebalance_date": style_meta.get(
            "latest_rebalance_date",
            industry_meta.get("latest_rebalance_date"),
        ),
        "latest_entry_date": style_meta.get(
            "latest_entry_date",
            industry_meta.get("latest_entry_date"),
        ),
        "style_factors": style_meta.get("factors", {}),
        "industry_column": industry_meta.get("industry_column"),
    }


def _auto_repair_breaches(
    exposure: Mapping[str, Any],
    *,
    cfg: Mapping[str, Any],
) -> pd.DataFrame:
    rows = []
    rows.extend(_auto_momentum_breaches(exposure.get("style"), cfg=cfg))
    rows.extend(_auto_bank_industry_breaches(exposure.get("industry"), cfg=cfg))
    return pd.DataFrame(
        rows,
        columns=[
            "status",
            "check",
            "rebalance_date",
            "entry_date",
            "name",
            "metric",
            "value",
            "limit",
        ],
    )


def _auto_momentum_breaches(
    style: object,
    *,
    cfg: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not isinstance(style, pd.DataFrame) or style.empty:
        return []
    limit = float(cfg.get("max_abs_momentum_active", 1.0))
    rows = []
    for _, row in style.loc[style["factor"].astype(str).eq("momentum")].iterrows():
        metric, value = _first_finite_metric(row, "active_net_vs_cap", "active_net_vs_equal")
        if value is None or abs(value) <= limit:
            continue
        rows.append(
            _breach_row(
                row,
                check="style_active",
                name="momentum",
                metric=metric,
                value=value,
                limit=limit,
            )
        )
    return rows


def _auto_bank_industry_breaches(
    industry: object,
    *,
    cfg: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not isinstance(industry, pd.DataFrame) or industry.empty:
        return []
    bank_name = str(cfg.get("bank_industry_name", "银行"))
    limit = float(cfg.get("max_abs_industry_active", 0.20))
    bank_rows = industry.loc[industry["industry"].astype(str).eq(bank_name)]
    rows = []
    for _, row in bank_rows.iterrows():
        metric, value = _first_finite_metric(
            row,
            "active_net_vs_cap_weight",
            "active_net_vs_equal_weight",
        )
        if value is None or abs(value) <= limit:
            continue
        rows.append(
            _breach_row(
                row,
                check="industry_active",
                name=bank_name,
                metric=metric,
                value=value,
                limit=limit,
            )
        )
    return rows


def _breach_row(
    row: pd.Series,
    *,
    check: str,
    name: str,
    metric: str,
    value: float,
    limit: float,
) -> dict[str, Any]:
    return {
        "status": "breached",
        "check": check,
        "rebalance_date": row.get("rebalance_date"),
        "entry_date": row.get("entry_date"),
        "name": name,
        "metric": metric,
        "value": float(value),
        "limit": float(limit),
    }


def _first_finite_metric(row: pd.Series, *columns: str) -> tuple[str, float | None]:
    for column in columns:
        value = row.get(column)
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            return column, number
    return columns[0], None


def _cfg_enabled(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return isinstance(value, Mapping) and bool(value.get("enabled", False))


def _as_mapping(value: object) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, bool):
        return {"enabled": value}
    return {}


def _load_table(path_value: str | Path) -> pd.DataFrame:
    path = resolve_data_input_path(str(path_value))
    if not path.exists():
        raise SystemExit(f"Configured postprocess file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix in {".json", ".jsonl"}:
        return pd.read_json(path, lines=suffix == ".jsonl")
    raise SystemExit(f"Unsupported postprocess file format: {path}")


def _first_path_value(cfg: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = cfg.get(key)
        if value:
            return str(value)
    return None


def _repair_config_from_mapping(cfg: Mapping[str, Any]) -> PostBufferExposureRepairConfig:
    field_names = {field.name for field in fields(PostBufferExposureRepairConfig)}
    payload = {key: value for key, value in cfg.items() if key in field_names}
    return PostBufferExposureRepairConfig(**payload)


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


def _series_from_position_backtest(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.Series(
        pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float),
        index=pd.to_datetime(frame["period_end"], errors="coerce"),
        name=column,
    )
