from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ._construction_grid_io import (
    _first_non_empty,
    _get_nested,
    _load_json,
    _load_pricing_data,
    _load_scored_data,
    _resolve_path,
    _resolve_rebalance_dates,
)
from .signal_postprocess import apply_score_postprocess


def _prepare_signal_column(
    data: pd.DataFrame,
    signal_col: str,
    variant: dict[str, Any],
    *,
    target_col: str,
    dynamic_ensemble_fn: Any | None,
) -> tuple[pd.DataFrame, str, str, str, dict[str, Any]]:
    ensemble = variant.get("dynamic_ensemble") or variant.get("dynamic_signal_ensemble")
    if ensemble is not None:
        if not isinstance(ensemble, dict):
            raise ValueError("dynamic_ensemble must be a mapping.")
        if dynamic_ensemble_fn is None:
            raise ValueError(
                "dynamic_ensemble requires an injected dynamic_ensemble_fn. "
                "Precompute dynamic ensemble scores in alpha-research or use the "
                "strategy-pipeline CLI."
            )
        out, ensemble_col, result = dynamic_ensemble_fn(
            data,
            spec=ensemble,
            target_col=target_col,
        )
        summary = getattr(result, "summary", {})
        if not isinstance(summary, dict):
            summary = {}
        method = "dynamic_ensemble"
        columns = ",".join(str(col) for col in ensemble.get("signal_cols", []))
        meta: dict[str, Any] = {
            "dynamic_ensemble_active": True,
            "dynamic_ensemble_signal_cols": columns,
            "dynamic_ensemble_avg_active_factor_count": summary.get("avg_active_factor_count"),
            "dynamic_ensemble_avg_factor_turnover": summary.get("avg_factor_turnover"),
            "dynamic_ensemble_avg_stock_turnover": summary.get("avg_stock_turnover"),
            "factor_correlation_threshold": summary.get("correlation_threshold"),
            "dynamic_ensemble_result": result,
        }
        data = out
        signal_col = ensemble_col
    else:
        meta = {
            "dynamic_ensemble_active": False,
            "dynamic_ensemble_signal_cols": None,
            "dynamic_ensemble_avg_active_factor_count": None,
            "dynamic_ensemble_avg_factor_turnover": None,
            "dynamic_ensemble_avg_stock_turnover": None,
            "factor_correlation_threshold": None,
        }
        method = "none"
        columns = ""

    postprocess = variant.get("score_postprocess") or {}
    if not isinstance(postprocess, dict):
        raise ValueError("score_postprocess must be a mapping.")
    postprocess_method = str(postprocess.get("method", "none")).strip().lower()
    postprocess_columns = [str(col) for col in postprocess.get("columns", [])]
    if postprocess_method != "none":
        out = data.copy()
        derived_col = f"__construction_score_{variant.get('name', 'variant')}"
        out[derived_col] = apply_score_postprocess(
            out,
            signal_col,
            method=postprocess_method,
            columns=postprocess_columns,
            strength=float(postprocess.get("strength", 1.0)),
            min_obs=postprocess.get("min_obs"),
        )
        data = out
        signal_col = derived_col
        method = postprocess_method if method == "none" else f"{method}+{postprocess_method}"
        columns = ",".join(postprocess_columns)

    risk_penalty = variant.get("risk_penalty") or {}
    if risk_penalty:
        if not isinstance(risk_penalty, dict):
            raise ValueError("risk_penalty must be a mapping.")
        risk_columns = [str(col) for col in risk_penalty.get("columns", [])]
        missing = [col for col in risk_columns if col not in data.columns]
        if missing:
            raise ValueError(f"Risk penalty columns not found: {', '.join(sorted(set(missing)))}")
        strength = float(risk_penalty.get("strength", risk_penalty.get("scale", 0.0)))
        adjusted_col = f"__risk_adjusted_score_{variant.get('name', 'variant')}"
        out = data.copy()
        if risk_columns and strength != 0.0:
            risk = out[risk_columns].apply(pd.to_numeric, errors="coerce")
            grouped = risk.groupby(out["trade_date"], sort=False)
            mean = grouped.transform("mean")
            std = grouped.transform(lambda series: series.std(ddof=0)).replace(0.0, np.nan)
            penalty = risk.sub(mean).div(std).abs().mean(axis=1).fillna(0.0)
            out[adjusted_col] = pd.to_numeric(out[signal_col], errors="coerce") - strength * penalty
        else:
            out[adjusted_col] = out[signal_col]
        data = out
        signal_col = adjusted_col
        meta["risk_penalty_columns"] = ",".join(risk_columns)
        meta["risk_penalty_strength"] = strength
        method = "risk_penalty" if method == "none" else f"{method}+risk_penalty"
    else:
        meta["risk_penalty_columns"] = None
        meta["risk_penalty_strength"] = None

    return data, signal_col, method, columns, meta


def _construction_grid_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = config.get("construction_grid", config)
    if not isinstance(cfg, dict):
        raise SystemExit("construction_grid must be a mapping.")
    return cfg


def _load_grid_data_inputs(cfg: dict[str, Any], *, config_dir: Path) -> dict[str, Any]:
    summary_path = _resolve_path(
        cfg.get("summary_file") or cfg.get("summary_path"), base_dir=config_dir
    )
    summary = _load_json(summary_path)
    run_dir = _resolve_path(_get_nested(summary, "run", "output_dir"), base_dir=config_dir)
    if run_dir is None and summary_path is not None:
        run_dir = summary_path.parent

    scored_file = _first_non_empty(
        cfg.get("scored_file"),
        _get_nested(summary, "eval", "scored_file"),
    )
    scored_path = _resolve_path(scored_file, base_dir=run_dir or config_dir)
    if scored_path is None:
        raise SystemExit("Construction grid requires scored_file or summary.eval.scored_file.")
    scored_data = _load_scored_data(scored_path)
    pricing_file = _first_non_empty(cfg.get("pricing_file"), cfg.get("backtest_pricing_file"))
    pricing_path = _resolve_path(pricing_file, base_dir=run_dir or config_dir)
    pricing_data = _load_pricing_data(pricing_path) if pricing_path is not None else scored_data
    return {
        "summary": summary,
        "summary_path": summary_path,
        "scored_file": scored_path,
        "pricing_file": pricing_path,
        "scored_data": scored_data,
        "pricing_data": pricing_data,
    }


def _resolve_grid_columns(
    cfg: dict[str, Any],
    summary: dict[str, Any],
    scored_data: pd.DataFrame,
) -> dict[str, str]:
    target_col = str(
        _first_non_empty(
            cfg.get("target_col"),
            _get_nested(summary, "label", "target_col"),
            "future_return",
        )
    )
    price_col = str(
        _first_non_empty(
            cfg.get("price_col"),
            _get_nested(summary, "data", "price_col"),
            "close",
        )
    )
    eval_signal_col = str(
        _first_non_empty(
            cfg.get("eval_signal_col"),
            _get_nested(summary, "eval", "scored_signal_col"),
            "signal_eval",
            "pred",
        )
    )
    if eval_signal_col not in scored_data.columns and "pred" in scored_data.columns:
        eval_signal_col = "pred"
    backtest_signal_col = str(
        _first_non_empty(
            cfg.get("backtest_signal_col"),
            _get_nested(summary, "eval", "scored_signal_backtest_col"),
            eval_signal_col,
        )
    )
    if backtest_signal_col not in scored_data.columns:
        backtest_signal_col = eval_signal_col

    required = ("trade_date", "symbol", target_col, price_col, eval_signal_col, backtest_signal_col)
    missing_cols = [col for col in required if col not in scored_data.columns]
    if missing_cols:
        raise SystemExit("Missing required columns in scored data: " + ", ".join(missing_cols))
    return {
        "target_col": target_col,
        "price_col": price_col,
        "eval_signal_col": eval_signal_col,
        "backtest_signal_col": backtest_signal_col,
    }


def _resolve_grid_rebalance_dates(
    cfg: dict[str, Any],
    summary: dict[str, Any],
    scored_data: pd.DataFrame,
) -> dict[str, list[pd.Timestamp]]:
    min_symbols_per_date = int(
        _first_non_empty(
            cfg.get("min_symbols_per_date"), _get_nested(summary, "data", "min_symbols_per_date"), 1
        )
    )
    eval_frequency = str(
        _first_non_empty(
            cfg.get("eval_rebalance_frequency"),
            cfg.get("rebalance_frequency"),
            _get_nested(summary, "eval", "rebalance_frequency"),
            "W",
        )
    )
    backtest_frequency = str(
        _first_non_empty(
            cfg.get("backtest_rebalance_frequency"),
            cfg.get("rebalance_frequency"),
            _get_nested(summary, "backtest", "rebalance_frequency"),
            eval_frequency,
        )
    )
    eval_rebalance_dates = _resolve_rebalance_dates(
        _first_non_empty(
            cfg.get("eval_rebalance_dates"),
            cfg.get("rebalance_dates"),
            _get_nested(summary, "eval", "rebalance_dates"),
        ),
        scored_data,
        eval_frequency,
        min_symbols_per_date,
    )
    backtest_rebalance_dates = _resolve_rebalance_dates(
        _first_non_empty(
            cfg.get("backtest_rebalance_dates"),
            cfg.get("rebalance_dates"),
            _get_nested(summary, "backtest", "rebalance_dates"),
        ),
        scored_data,
        backtest_frequency,
        min_symbols_per_date,
    )
    return {
        "eval_rebalance_dates": eval_rebalance_dates,
        "backtest_rebalance_dates": backtest_rebalance_dates,
    }


def _validated_variants(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    variants = cfg.get("variants")
    if not isinstance(variants, list) or not variants:
        raise SystemExit("construction_grid.variants must be a non-empty list.")
    for idx, variant in enumerate(variants, start=1):
        if not isinstance(variant, dict):
            raise SystemExit(f"construction_grid.variants[{idx}] must be a mapping.")
    return variants


def _build_base_context(config: dict[str, Any], config_dir: Path) -> dict[str, Any]:
    cfg = _construction_grid_config(config)
    data_inputs = _load_grid_data_inputs(cfg, config_dir=config_dir)
    summary = data_inputs["summary"]
    scored_data = data_inputs["scored_data"]
    columns = _resolve_grid_columns(cfg, summary, scored_data)
    rebalance_dates = _resolve_grid_rebalance_dates(cfg, summary, scored_data)

    return {
        "cfg": cfg,
        "summary": summary,
        "summary_path": data_inputs["summary_path"],
        "scored_file": data_inputs["scored_file"],
        "pricing_file": data_inputs["pricing_file"],
        "scored_data": scored_data,
        "pricing_data": data_inputs["pricing_data"],
        "target_col": columns["target_col"],
        "price_col": columns["price_col"],
        "eval_signal_col": columns["eval_signal_col"],
        "backtest_signal_col": columns["backtest_signal_col"],
        "eval_rebalance_dates": rebalance_dates["eval_rebalance_dates"],
        "backtest_rebalance_dates": rebalance_dates["backtest_rebalance_dates"],
        "variants": _validated_variants(cfg),
    }
