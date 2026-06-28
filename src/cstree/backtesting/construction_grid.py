from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from market_data_platform.symbols import canonicalize_symbol_columns

from . import construction_grid_reports as _construction_grid_reports
from .benchmarking import build_benchmark_series
from .execution import build_execution_model
from .metrics import (
    daily_ic_series,
    estimate_turnover,
    quantile_returns,
    summarize_active_returns,
    summarize_ic,
)
from .rebalance import get_rebalance_dates
from .signal_postprocess import apply_score_postprocess

build_inertia_selection_report = _construction_grid_reports.build_inertia_selection_report
select_construction_variant_with_inertia = (
    _construction_grid_reports.select_construction_variant_with_inertia
)
write_reports = _construction_grid_reports.write_reports

BacktestTopKFn = Callable[..., Any]
DynamicEnsembleFn = Callable[..., tuple[pd.DataFrame, str, Any]]


def _resolve_path(path_text: str | Path | None, *, base_dir: Path | None = None) -> Path | None:
    if path_text is None:
        return None
    candidate = Path(path_text).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    if base_dir is not None:
        by_base = (base_dir / candidate).resolve()
        if by_base.exists():
            return by_base
    return (Path.cwd() / candidate).resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Construction grid config not found: {path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"Failed to parse construction grid config: {path} ({exc})") from exc
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise SystemExit(f"Construction grid config must be a mapping: {path}")
    return payload


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"Failed to parse summary JSON: {path} ({exc})") from exc
    return payload if isinstance(payload, dict) else {}


def _get_nested(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _first_non_empty(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _coerce_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def _periods_per_year(stats: dict[str, Any], fallback: int) -> float:
    value = stats.get("periods_per_year")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    return number if np.isfinite(number) and number > 0 else float(fallback)


def _parse_date_series(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.strip()
    date8 = text.str.fullmatch(r"\d{8}")
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    if date8.any():
        parsed.loc[date8] = pd.to_datetime(text.loc[date8], format="%Y%m%d", errors="coerce")
    if (~date8).any():
        parsed.loc[~date8] = pd.to_datetime(text.loc[~date8], errors="coerce")
    return parsed


def _read_returns_file(path: Path) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(f"Benchmark returns file not found: {path}")
    frame = pd.read_csv(path)
    date_col = next(
        (col for col in ("trade_date", "date", "period_end") if col in frame.columns), None
    )
    ret_col = next(
        (
            col
            for col in (
                "benchmark_return",
                "return",
                "net_return",
                "strategy_return",
                "active_return",
            )
            if col in frame.columns
        ),
        None,
    )
    if date_col is None or ret_col is None:
        raise ValueError(
            "Returns file must include a date column and one return column "
            "(benchmark_return, return, net_return, or strategy_return)."
        )
    series = pd.Series(
        pd.to_numeric(frame[ret_col], errors="coerce").to_numpy(dtype=float),
        index=_parse_date_series(frame[date_col]),
        name=ret_col,
    ).dropna()
    return series.sort_index()


def _parse_date_list(values: Any) -> list[pd.Timestamp]:
    if not isinstance(values, list):
        return []
    parsed: list[pd.Timestamp] = []
    for raw in values:
        dt = pd.to_datetime(raw, format="%Y%m%d", errors="coerce")
        if pd.isna(dt):
            dt = pd.to_datetime(raw, errors="coerce")
        if not pd.isna(dt):
            parsed.append(pd.Timestamp(dt))
    return sorted(dict.fromkeys(parsed))


def _resolve_rebalance_dates(
    summary_dates: Any,
    scored_data: pd.DataFrame,
    frequency: str,
    min_symbols_per_date: int,
) -> list[pd.Timestamp]:
    parsed = _parse_date_list(summary_dates)
    available = set(pd.to_datetime(scored_data["trade_date"].unique()))
    if parsed:
        return [date for date in parsed if date in available]

    trade_dates = sorted(available)
    dates = get_rebalance_dates(trade_dates, frequency)
    if min_symbols_per_date > 1:
        counts = scored_data.groupby("trade_date")["symbol"].nunique()
        valid_dates = set(pd.to_datetime(counts[counts >= min_symbols_per_date].index))
        dates = [date for date in dates if date in valid_dates]
    return dates


def _load_scored_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Scored file not found: {path}")
    frame = pd.read_parquet(path)
    if frame.empty:
        raise SystemExit(f"Scored file is empty: {path}")
    frame = canonicalize_symbol_columns(frame, context="Construction grid scored data")
    if "trade_date" not in frame.columns:
        raise SystemExit("Scored data must include trade_date.")
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    return frame


def _load_pricing_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Pricing file not found: {path}")
    frame = pd.read_parquet(path)
    if frame.empty:
        raise SystemExit(f"Pricing file is empty: {path}")
    frame = canonicalize_symbol_columns(frame, context="Construction grid pricing data")
    missing = [col for col in ("trade_date", "symbol") if col not in frame.columns]
    if missing:
        raise SystemExit("Pricing data must include: " + ", ".join(missing))
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame["symbol"] = frame["symbol"].astype(str)
    return frame.sort_values(["trade_date", "symbol"]).reset_index(drop=True)


def _prepare_signal_column(
    data: pd.DataFrame,
    signal_col: str,
    variant: dict[str, Any],
    *,
    target_col: str,
    dynamic_ensemble_fn: DynamicEnsembleFn | None,
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
        meta = {
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


def _init_row(
    *,
    variant: dict[str, Any],
    context: dict[str, Any],
    signal_col: str,
    score_postprocess_method: str,
    score_postprocess_columns: str,
) -> dict[str, Any]:
    cfg = context["cfg"]
    summary = context["summary"]
    top_k = int(
        _first_non_empty(
            variant.get("top_k"), cfg.get("top_k"), _get_nested(summary, "backtest", "top_k"), 10
        )
    )
    long_only = _coerce_bool(
        _first_non_empty(
            variant.get("long_only"),
            cfg.get("long_only"),
            _get_nested(summary, "backtest", "long_only"),
        ),
        default=True,
    )
    short_k_raw = _first_non_empty(
        variant.get("short_k"), cfg.get("short_k"), _get_nested(summary, "backtest", "short_k")
    )
    short_k = int(short_k_raw) if short_k_raw is not None else None
    cost_bps = float(
        _first_non_empty(
            variant.get("cost_bps"),
            variant.get("transaction_cost_bps"),
            cfg.get("cost_bps"),
            cfg.get("transaction_cost_bps"),
            _get_nested(summary, "backtest", "transaction_cost_bps"),
            0.0,
        )
    )
    benchmark_name = _first_non_empty(variant.get("benchmark_name"), cfg.get("benchmark_name"))
    benchmark_returns_file = _first_non_empty(
        variant.get("benchmark_returns_file"),
        cfg.get("benchmark_returns_file"),
    )
    return {
        "variant": str(variant.get("name") or f"k{top_k}_bps{cost_bps:g}"),
        "scored_file": str(context["scored_file"]),
        "summary_path": str(context["summary_path"]) if context["summary_path"] else None,
        "target_col": context["target_col"],
        "price_col": context["price_col"],
        "eval_signal_col": signal_col,
        "backtest_signal_col": signal_col,
        "top_k": top_k,
        "rank_offset": int(_first_non_empty(variant.get("rank_offset"), cfg.get("rank_offset"), 0)),
        "short_k": short_k,
        "long_only": long_only,
        "cost_bps": cost_bps,
        "buffer_exit": int(_first_non_empty(variant.get("buffer_exit"), cfg.get("buffer_exit"), 0)),
        "buffer_entry": int(
            _first_non_empty(variant.get("buffer_entry"), cfg.get("buffer_entry"), 0)
        ),
        "weighting": str(
            _first_non_empty(variant.get("weighting"), cfg.get("weighting"), "equal")
        ).lower(),
        "weighting_liquidity_col": str(
            _first_non_empty(
                variant.get("weighting_liquidity_col"),
                cfg.get("weighting_liquidity_col"),
                "medadv20_amount",
            )
        ),
        "liquidity_floor_col": _first_non_empty(
            variant.get("liquidity_floor_col"),
            cfg.get("liquidity_floor_col"),
        ),
        "liquidity_floor_quantile": _first_non_empty(
            variant.get("liquidity_floor_quantile"),
            cfg.get("liquidity_floor_quantile"),
        ),
        "max_turnover_per_rebalance": _first_non_empty(
            variant.get("max_turnover_per_rebalance"),
            cfg.get("max_turnover_per_rebalance"),
        ),
        "score_postprocess_method": score_postprocess_method,
        "score_postprocess_columns": score_postprocess_columns,
        "benchmark_name": str(benchmark_name) if benchmark_name is not None else None,
        "benchmark_returns_file": (
            str(benchmark_returns_file) if benchmark_returns_file is not None else None
        ),
        "exposure_available": False,
        "status": "ok",
        "error": None,
    }


def _variant_validation_error(row: dict[str, Any]) -> str | None:
    if row["top_k"] <= 0:
        return "top_k must be positive."
    if int(row["rank_offset"]) < 0:
        return "rank_offset must be >= 0."
    if row["short_k"] is not None and int(row["short_k"]) < 0:
        return "short_k must be >= 0."
    if row["weighting"] not in {"equal", "signal", "sqrt_liquidity"}:
        return "weighting must be one of: equal, signal, sqrt_liquidity."
    return None


def _mark_failed(row: dict[str, Any], error: str) -> dict[str, Any]:
    row["status"] = "failed"
    row["error"] = error
    return row


def _update_eval_metrics(
    row: dict[str, Any],
    *,
    context: dict[str, Any],
    variant: dict[str, Any],
    data: pd.DataFrame,
    signal_col: str,
) -> None:
    cfg = context["cfg"]
    target_col = context["target_col"]
    eval_slice = data[data["trade_date"].isin(context["eval_rebalance_dates"])].copy()
    ic_stats = summarize_ic(daily_ic_series(eval_slice, target_col, signal_col))
    row["eval_ic_mean"] = ic_stats.get("mean")
    row["eval_ic_ir"] = ic_stats.get("ir")

    n_quantiles = int(_first_non_empty(variant.get("n_quantiles"), cfg.get("n_quantiles"), 5))
    quantile_ts = quantile_returns(eval_slice, signal_col, target_col, n_quantiles)
    quantile_mean = quantile_ts.mean() if not quantile_ts.empty else pd.Series(dtype=float)
    row["eval_long_short"] = (
        float(quantile_mean.iloc[-1] - quantile_mean.iloc[0]) if not quantile_mean.empty else None
    )

    if not context["eval_rebalance_dates"]:
        return
    turnover = estimate_turnover(
        eval_slice,
        signal_col,
        int(row["top_k"]),
        context["eval_rebalance_dates"],
        buffer_exit=int(row["buffer_exit"]),
        buffer_entry=int(row["buffer_entry"]),
        rank_offset=int(row["rank_offset"]),
    )
    row["eval_turnover_mean"] = float(turnover.mean()) if not turnover.empty else None


def _optional_existing_column(value: Any, data: pd.DataFrame) -> str | None:
    column = str(value) if value is not None else None
    if column and column in data.columns:
        return column
    return None


def _build_variant_backtest_options(
    row: dict[str, Any],
    *,
    context: dict[str, Any],
    variant: dict[str, Any],
    data: pd.DataFrame,
) -> dict[str, Any]:
    cfg = context["cfg"]
    summary = context["summary"]
    price_col = context["price_col"]
    execution_cfg = _first_non_empty(variant.get("execution"), cfg.get("execution"))
    exit_price_policy = str(
        _first_non_empty(
            variant.get("exit_price_policy"),
            cfg.get("exit_price_policy"),
            _get_nested(summary, "backtest", "exit_price_policy"),
            "strict",
        )
    ).lower()
    exit_fallback_policy = str(
        _first_non_empty(
            variant.get("exit_fallback_policy"),
            cfg.get("exit_fallback_policy"),
            _get_nested(summary, "backtest", "exit_fallback_policy"),
            "ffill",
        )
    ).lower()
    label_horizon = _first_non_empty(
        variant.get("exit_horizon_days"),
        cfg.get("exit_horizon_days"),
        _get_nested(summary, "backtest", "exit_horizon_days"),
        _get_nested(summary, "label", "horizon_days"),
    )
    tradable_col = _optional_existing_column(
        _first_non_empty(
            variant.get("tradable_col"),
            cfg.get("tradable_col"),
            _get_nested(summary, "backtest", "tradable_col"),
            "is_tradable",
        ),
        data,
    )
    group_col = _optional_existing_column(
        _first_non_empty(
            variant.get("group_col"),
            cfg.get("group_col"),
            _get_nested(summary, "backtest", "group_col"),
        ),
        data,
    )
    row["exposure_available"] = bool(group_col)

    max_names_per_group = _first_non_empty(
        variant.get("max_names_per_group"),
        cfg.get("max_names_per_group"),
        _get_nested(summary, "backtest", "max_names_per_group"),
    )
    trading_days = int(
        _first_non_empty(
            variant.get("trading_days_per_year"),
            cfg.get("trading_days_per_year"),
            _get_nested(summary, "backtest", "trading_days_per_year"),
            252,
        )
    )
    return {
        "exit_price_policy": exit_price_policy,
        "exit_fallback_policy": exit_fallback_policy,
        "execution_model": build_execution_model(
            execution_cfg,
            default_cost_bps=float(row["cost_bps"]),
            default_exit_price_policy=exit_price_policy,
            default_exit_fallback_policy=exit_fallback_policy,
            default_price_col=price_col,
        ),
        "exit_horizon_days": int(label_horizon) if label_horizon is not None else None,
        "tradable_col": tradable_col,
        "group_col": group_col,
        "max_names_per_group": (
            int(max_names_per_group) if max_names_per_group is not None else None
        ),
        "trading_days": trading_days,
        "shift_days": int(
            _first_non_empty(
                variant.get("shift_days"),
                cfg.get("shift_days"),
                _get_nested(summary, "label", "shift_days"),
                0,
            )
        ),
        "exit_mode": str(
            _first_non_empty(variant.get("exit_mode"), cfg.get("exit_mode"), "rebalance")
        ).lower(),
    }


def _run_variant_backtest(
    row: dict[str, Any],
    *,
    context: dict[str, Any],
    variant: dict[str, Any],
    data: pd.DataFrame,
    signal_col: str,
    backtest_topk_fn: BacktestTopKFn,
) -> tuple[Any, int]:
    options = _build_variant_backtest_options(row, context=context, variant=variant, data=data)
    result = backtest_topk_fn(
        data,
        pred_col=signal_col,
        price_col=context["price_col"],
        rebalance_dates=context["backtest_rebalance_dates"],
        top_k=int(row["top_k"]),
        rank_offset=int(row["rank_offset"]),
        shift_days=options["shift_days"],
        cost_bps=float(row["cost_bps"]),
        trading_days_per_year=options["trading_days"],
        exit_mode=options["exit_mode"],
        exit_horizon_days=options["exit_horizon_days"],
        long_only=bool(row["long_only"]),
        short_k=row["short_k"],
        weighting=str(row["weighting"]),
        buffer_exit=int(row["buffer_exit"]),
        buffer_entry=int(row["buffer_entry"]),
        liquidity_floor_col=(
            str(row["liquidity_floor_col"]) if row["liquidity_floor_col"] is not None else None
        ),
        liquidity_floor_quantile=(
            float(row["liquidity_floor_quantile"])
            if row["liquidity_floor_quantile"] is not None
            else None
        ),
        weighting_liquidity_col=str(row["weighting_liquidity_col"]),
        max_turnover_per_rebalance=(
            float(row["max_turnover_per_rebalance"])
            if row["max_turnover_per_rebalance"] is not None
            else None
        ),
        tradable_col=options["tradable_col"],
        group_col=options["group_col"],
        max_names_per_group=options["max_names_per_group"],
        exit_price_policy=options["exit_price_policy"],
        exit_fallback_policy=options["exit_fallback_policy"],
        execution=options["execution_model"],
        pricing_data=context["pricing_data"],
    )
    return result, int(options["trading_days"])


def _update_backtest_metrics(
    row: dict[str, Any],
    *,
    bt_stats: dict[str, Any],
    gross_series: pd.Series,
) -> None:
    row["backtest_periods"] = bt_stats.get("periods")
    row["backtest_total_return"] = bt_stats.get("total_return")
    row["backtest_gross_total_return"] = float((1.0 + gross_series).prod() - 1.0)
    row["backtest_ann_return"] = bt_stats.get("ann_return")
    row["backtest_ann_vol"] = bt_stats.get("ann_vol")
    row["backtest_sharpe"] = bt_stats.get("sharpe")
    row["backtest_max_drawdown"] = bt_stats.get("max_drawdown")
    row["backtest_avg_turnover"] = bt_stats.get("avg_turnover")
    row["backtest_avg_cost_drag"] = bt_stats.get("avg_cost_drag")


def _update_active_metrics(
    row: dict[str, Any],
    *,
    context: dict[str, Any],
    bt_stats: dict[str, Any],
    net_series: pd.Series,
    period_info: Any,
    trading_days: int,
) -> None:
    benchmark_path = _resolve_path(
        row.get("benchmark_returns_file") or None,
        base_dir=Path(str(context["scored_file"])).parent,
    )
    if not benchmark_path:
        return
    benchmark = _read_returns_file(benchmark_path)
    benchmark_series, _ = build_benchmark_series(
        None,
        context["price_col"],
        context["price_col"],
        period_info,
        benchmark_return_series=benchmark,
    )
    active_stats, _ = summarize_active_returns(
        net_series,
        benchmark_series,
        periods_per_year=_periods_per_year(bt_stats, trading_days),
    )
    row["active_total_return"] = active_stats.get("active_total_return")
    row["information_ratio"] = active_stats.get("information_ratio")
    row["tracking_error"] = active_stats.get("tracking_error")
    row["beta"] = active_stats.get("beta")
    row["alpha"] = active_stats.get("alpha")
    row["corr"] = active_stats.get("corr")


def _evaluate_variant(
    context: dict[str, Any],
    variant: dict[str, Any],
    *,
    backtest_topk_fn: BacktestTopKFn,
    dynamic_ensemble_fn: DynamicEnsembleFn | None,
) -> dict[str, Any]:
    data, signal_col, method, columns, signal_meta = _prepare_signal_column(
        context["scored_data"],
        context["backtest_signal_col"],
        variant,
        target_col=context["target_col"],
        dynamic_ensemble_fn=dynamic_ensemble_fn,
    )
    row = _init_row(
        variant=variant,
        context=context,
        signal_col=signal_col,
        score_postprocess_method=method,
        score_postprocess_columns=columns,
    )
    row.update(
        {key: value for key, value in signal_meta.items() if key != "dynamic_ensemble_result"}
    )
    validation_error = _variant_validation_error(row)
    if validation_error is not None:
        return _mark_failed(row, validation_error)

    try:
        _update_eval_metrics(
            row,
            context=context,
            variant=variant,
            data=data,
            signal_col=signal_col,
        )
        bt_result, trading_days = _run_variant_backtest(
            row,
            context=context,
            variant=variant,
            data=data,
            signal_col=signal_col,
            backtest_topk_fn=backtest_topk_fn,
        )
        if bt_result is None:
            row["status"] = "no_backtest"
            return row
        bt_stats, net_series, gross_series, _, period_info = bt_result
        _update_backtest_metrics(row, bt_stats=bt_stats, gross_series=gross_series)
        _update_active_metrics(
            row,
            context=context,
            bt_stats=bt_stats,
            net_series=net_series,
            period_info=period_info,
            trading_days=trading_days,
        )
    except Exception as exc:
        _mark_failed(row, str(exc))
    return row


def _resolve_backtest_topk_fn(candidate: Any) -> BacktestTopKFn:
    if candidate is None:
        raise SystemExit(
            "Construction grid requires an injected backtest_topk_fn. "
            "Use the cstree CLI or pass cstree.backtesting.engine.backtest_topk explicitly."
        )
    if not callable(candidate):
        raise SystemExit("Construction grid backtest_topk_fn must be callable.")
    return candidate


def build_construction_grid(
    config: dict[str, Any],
    *,
    config_dir: Path,
    backtest_topk_fn: BacktestTopKFn | None = None,
    dynamic_ensemble_fn: DynamicEnsembleFn | None = None,
) -> list[dict[str, Any]]:
    context = _build_base_context(config, config_dir)
    runner = _resolve_backtest_topk_fn(backtest_topk_fn)
    return [
        _evaluate_variant(
            context,
            variant,
            backtest_topk_fn=runner,
            dynamic_ensemble_fn=dynamic_ensemble_fn,
        )
        for variant in context["variants"]
    ]


def add_construction_grid_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", required=True, help="Construction grid YAML config.")
    parser.add_argument("--output", default=None, help="Output CSV path.")
    parser.add_argument("--output-json", default=None, help="Output JSON path.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging level",
    )
    return parser


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(levelname)s: %(message)s",
    )
    config_path = _resolve_path(args.config)
    assert config_path is not None
    config = _load_yaml(config_path)
    rows = build_construction_grid(
        config,
        config_dir=config_path.parent,
        backtest_topk_fn=_resolve_backtest_topk_fn(getattr(args, "backtest_topk_fn", None)),
        dynamic_ensemble_fn=getattr(args, "dynamic_ensemble_fn", None),
    )
    cfg = config.get("construction_grid", config)
    output_csv = _resolve_path(
        args.output or cfg.get("output_csv") or cfg.get("output"), base_dir=config_path.parent
    )
    output_json = _resolve_path(
        args.output_json or cfg.get("output_json"), base_dir=config_path.parent
    )
    selection_cfg = cfg.get("rolling_selection") or cfg.get("inertia_selection")
    selection_report = None
    selection_output = None
    if selection_cfg:
        if not isinstance(selection_cfg, dict):
            raise SystemExit("construction_grid.rolling_selection must be a mapping.")
        selection_report = build_inertia_selection_report(rows, selection_cfg)
        selection_output = _resolve_path(
            selection_cfg.get("output_json") or selection_cfg.get("output"),
            base_dir=config_path.parent,
        )
    if output_csv is None and output_json is None:
        print(json.dumps(rows, ensure_ascii=True, indent=2, default=str))
    else:
        write_reports(rows, output_csv=output_csv, output_json=output_json)
        if output_csv:
            logging.info("Construction grid CSV written to %s", output_csv)
        if output_json:
            logging.info("Construction grid JSON written to %s", output_json)
    if selection_report is not None:
        if selection_output is None:
            print(json.dumps(selection_report, ensure_ascii=True, indent=2, default=str))
        else:
            selection_output.parent.mkdir(parents=True, exist_ok=True)
            selection_output.write_text(
                json.dumps(selection_report, ensure_ascii=True, indent=2, default=str),
                encoding="utf-8",
            )
            logging.info("Construction grid rolling selection JSON written to %s", selection_output)
    return rows
