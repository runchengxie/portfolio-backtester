"""Position post-processing: post-buffer exposure repair and backtest rebuild."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import fields
from typing import Any

import numpy as np
import pandas as pd

from portfolio_backtester._symbol_utils import resolve_data_input_path

from ._position_postprocess_overlay import _apply_cash_gross_overlay
from ._postprocess_shared import _as_mapping, _cfg_enabled, _first_path_value, _load_table
from .exposure import compute_backtest_exposure_analysis
from .position_backtest import PositionBacktestConfig, run_position_backtest
from .post_buffer_exposure_repair import (
    PostBufferExposureRepairConfig,
    repair_post_buffer_exposure,
)

logger = logging.getLogger("portfolio_backtester")


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


def _repair_config_from_mapping(cfg: Mapping[str, Any]) -> PostBufferExposureRepairConfig:
    field_names = {field.name for field in fields(PostBufferExposureRepairConfig)}
    payload = {key: value for key, value in cfg.items() if key in field_names}
    return PostBufferExposureRepairConfig(**payload)


def _series_from_position_backtest(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.Series(
        pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float),
        index=pd.to_datetime(frame["period_end"], errors="coerce"),
        name=column,
    )
