"""Shared normalization for portfolio backtest configuration.

The module owns configuration semantics that do not depend on a data provider or
strategy-pipeline runtime.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, SupportsInt, cast


def _normalize_positive_int_or_none(raw_value: object | None, *, error_message: str) -> int | None:
    if raw_value is None:
        return None
    try:
        value = int(cast("SupportsInt", raw_value))
    except (TypeError, ValueError) as exc:
        raise SystemExit(error_message) from exc
    if value <= 0:
        raise SystemExit(error_message)
    return value


def _normalize_optional_text(raw_value: object | None) -> str | None:
    if raw_value is None:
        return None
    return str(raw_value).strip() or None


def _normalize_backtest_tearsheet_enabled(backtest_cfg: Mapping[str, Any]) -> bool:
    raw_value = backtest_cfg.get("tearsheet")
    if raw_value is None:
        raw_value = backtest_cfg.get("tearsheet_enabled", False)
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, Mapping):
        return bool(raw_value.get("enabled", False))
    if raw_value is None:
        return False
    raise SystemExit("backtest.tearsheet must be a boolean or a mapping with enabled.")


def _normalize_backtest_postprocess_config(
    backtest_cfg: Mapping[str, Any],
    *keys: str,
) -> dict[str, Any]:
    raw_value: object | None = None
    for key in keys:
        if key in backtest_cfg:
            raw_value = backtest_cfg.get(key)
            break
    if raw_value is None:
        return {"enabled": False}
    if isinstance(raw_value, bool):
        return {"enabled": raw_value}
    if isinstance(raw_value, Mapping):
        normalized = dict(raw_value)
        normalized["enabled"] = bool(normalized.get("enabled", True))
        return normalized
    joined = ", ".join(f"backtest.{key}" for key in keys)
    raise SystemExit(f"{joined} must be a boolean or a mapping with enabled.")


def _resolve_backtest_postprocess_settings(backtest_cfg: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "BACKTEST_POST_BUFFER_EXPOSURE_REPAIR": _normalize_backtest_postprocess_config(
            backtest_cfg,
            "post_buffer_exposure_repair",
        ),
        "BACKTEST_CASH_GROSS_OVERLAY": _normalize_backtest_postprocess_config(
            backtest_cfg,
            "cash_gross_overlay",
            "gross_overlay",
        ),
        "BACKTEST_FRESHNESS_OVERLAY": _normalize_backtest_postprocess_config(
            backtest_cfg,
            "freshness_overlay",
        ),
        "BACKTEST_PRESERVE_GROSS_EXPOSURE": bool(
            backtest_cfg.get("preserve_gross_exposure", False)
        ),
    }


def _resolve_backtest_benchmark_settings(
    backtest_cfg: Mapping[str, Any],
) -> tuple[str | None, str | None]:
    backtest_benchmark = backtest_cfg.get("benchmark_symbol")
    if backtest_benchmark is not None:
        backtest_benchmark = str(backtest_benchmark).strip() or None
    backtest_benchmark_returns_file = backtest_cfg.get("benchmark_returns_file")
    if backtest_benchmark_returns_file is not None:
        backtest_benchmark_returns_file = str(backtest_benchmark_returns_file).strip() or None
    if backtest_benchmark and backtest_benchmark_returns_file:
        raise SystemExit(
            "backtest.benchmark_symbol and backtest.benchmark_returns_file are mutually exclusive."
        )
    return backtest_benchmark, backtest_benchmark_returns_file


def _resolve_backtest_weighting(backtest_cfg: Mapping[str, Any]) -> str:
    backtest_weighting = str(backtest_cfg.get("weighting", "equal")).strip().lower()
    if backtest_weighting not in {"equal", "signal", "sqrt_liquidity"}:
        raise SystemExit("backtest.weighting must be one of: equal, signal, sqrt_liquidity.")
    return backtest_weighting


def _resolve_selection_score_settings(
    backtest_cfg: Mapping[str, Any],
) -> tuple[float | None, float | None, int | None]:
    raw_bucket_size = backtest_cfg.get("selection_score_bucket_size")
    selection_score_bucket_size = float(raw_bucket_size) if raw_bucket_size is not None else None
    if selection_score_bucket_size is not None and selection_score_bucket_size <= 0:
        raise SystemExit("backtest.selection_score_bucket_size must be > 0.")
    raw_margin = backtest_cfg.get("selection_score_margin")
    selection_score_margin = float(raw_margin) if raw_margin is not None else None
    if selection_score_margin is not None and selection_score_margin <= 0:
        raise SystemExit("backtest.selection_score_margin must be > 0.")
    selection_score_margin_rank_limit = _normalize_positive_int_or_none(
        backtest_cfg.get("selection_score_margin_rank_limit"),
        error_message="backtest.selection_score_margin_rank_limit must be a positive integer.",
    )
    return (
        selection_score_bucket_size,
        selection_score_margin,
        selection_score_margin_rank_limit,
    )


def _resolve_backtest_group_col(backtest_cfg: Mapping[str, Any]) -> str | None:
    backtest_group_col = backtest_cfg.get("group_col")
    if backtest_group_col is not None:
        return str(backtest_group_col).strip() or None
    return None


def _resolve_backtest_signal_direction(backtest_cfg: Mapping[str, Any]) -> float | None:
    backtest_signal_direction_raw = backtest_cfg.get("signal_direction")
    if backtest_signal_direction_raw is None:
        return None
    backtest_signal_direction = float(backtest_signal_direction_raw)
    if backtest_signal_direction == 0:
        raise SystemExit("backtest.signal_direction cannot be 0.")
    return backtest_signal_direction


def _resolve_backtest_exit_settings(
    backtest_cfg: Mapping[str, Any],
    *,
    label_horizon_days: int,
) -> tuple[str, int | None, str, str]:
    backtest_exit_mode = str(backtest_cfg.get("exit_mode", "rebalance")).strip().lower()
    if backtest_exit_mode not in {"rebalance", "label_horizon"}:
        raise SystemExit("backtest.exit_mode must be one of: rebalance, label_horizon.")
    backtest_exit_horizon_days = backtest_cfg.get("exit_horizon_days")
    if backtest_exit_mode == "label_horizon":
        if backtest_exit_horizon_days is None:
            backtest_exit_horizon_days = label_horizon_days
        backtest_exit_horizon_days = int(backtest_exit_horizon_days)
    return (
        backtest_exit_mode,
        backtest_exit_horizon_days,
        str(backtest_cfg.get("exit_price_policy", "strict")).strip().lower(),
        str(backtest_cfg.get("exit_fallback_policy", "ffill")).strip().lower(),
    )


def _normalize_benchmark_compare(raw_value: object | None) -> list[dict[str, str]]:
    if raw_value is None:
        return []
    if not isinstance(raw_value, (list, tuple)):
        raise SystemExit("backtest.benchmark_compare must be a list of compare benchmark specs.")

    normalized: list[dict[str, str]] = []
    for idx, item in enumerate(raw_value):
        name: str | None = None
        returns_file: str | None = None
        symbol: str | None = None
        if isinstance(item, str):
            returns_file = str(item).strip() or None
        elif isinstance(item, Mapping):
            name_raw = item.get("name")
            if name_raw is not None:
                name = str(name_raw).strip() or None
            returns_file_raw = item.get("returns_file") or item.get("file") or item.get("path")
            if returns_file_raw is not None:
                returns_file = str(returns_file_raw).strip() or None
            symbol_raw = item.get("symbol") or item.get("benchmark_symbol")
            if symbol_raw is not None:
                symbol = str(symbol_raw).strip() or None
        else:
            raise SystemExit(
                "backtest.benchmark_compare entries must be either strings or mappings."
            )

        if bool(returns_file) == bool(symbol):
            raise SystemExit(
                f"backtest.benchmark_compare[{idx}] must provide exactly one of "
                "returns_file or symbol."
            )
        if not name:
            name = Path(returns_file).stem if returns_file else str(symbol)
        spec = {"name": name}
        if returns_file:
            spec["source_type"] = "returns_file"
            spec["returns_file"] = returns_file
        else:
            spec["source_type"] = "symbol"
            spec["symbol"] = str(symbol)
        normalized.append(spec)
    return normalized


def resolve_backtest_base_settings(
    backtest_cfg: Mapping[str, Any],
    *,
    eval_top_k: int,
    eval_rebalance_frequency: str,
    eval_transaction_cost_bps: float,
    label_horizon_days: int,
) -> dict[str, Any]:
    backtest_benchmark, backtest_benchmark_returns_file = _resolve_backtest_benchmark_settings(
        backtest_cfg
    )
    backtest_weighting = _resolve_backtest_weighting(backtest_cfg)
    (
        selection_score_bucket_size,
        selection_score_margin,
        selection_score_margin_rank_limit,
    ) = _resolve_selection_score_settings(backtest_cfg)
    (
        backtest_exit_mode,
        backtest_exit_horizon_days,
        backtest_exit_price_policy,
        backtest_exit_fallback_policy,
    ) = _resolve_backtest_exit_settings(
        backtest_cfg,
        label_horizon_days=label_horizon_days,
    )

    return {
        "BACKTEST_ENABLED": bool(backtest_cfg.get("enabled", True)),
        "BACKTEST_TOP_K": int(backtest_cfg.get("top_k", eval_top_k)),
        "BACKTEST_REBALANCE_FREQUENCY": backtest_cfg.get(
            "rebalance_frequency",
            eval_rebalance_frequency,
        ),
        "BACKTEST_COST_BPS": float(
            backtest_cfg.get("transaction_cost_bps", eval_transaction_cost_bps)
        ),
        "BACKTEST_TRADING_DAYS_PER_YEAR": int(backtest_cfg.get("trading_days_per_year", 252)),
        "BACKTEST_BENCHMARK": backtest_benchmark,
        "BACKTEST_BENCHMARK_RETURNS_FILE": backtest_benchmark_returns_file,
        "BACKTEST_BENCHMARK_COMPARE": _normalize_benchmark_compare(
            backtest_cfg.get("benchmark_compare")
        ),
        "BACKTEST_TEARSHEET_ENABLED": _normalize_backtest_tearsheet_enabled(backtest_cfg),
        "BACKTEST_LONG_ONLY": bool(backtest_cfg.get("long_only", True)),
        "BACKTEST_BUFFER_EXIT": int(backtest_cfg.get("buffer_exit", 0)),
        "BACKTEST_BUFFER_ENTRY": int(backtest_cfg.get("buffer_entry", 0)),
        "BACKTEST_WEIGHTING": backtest_weighting,
        "BACKTEST_GROUP_COL": _resolve_backtest_group_col(backtest_cfg),
        "BACKTEST_MAX_NAMES_PER_GROUP": _normalize_positive_int_or_none(
            backtest_cfg.get("max_names_per_group"),
            error_message="backtest.max_names_per_group must be a positive integer.",
        ),
        "BACKTEST_LIQUIDITY_FLOOR_COL": _normalize_optional_text(
            backtest_cfg.get("liquidity_floor_col")
        ),
        "BACKTEST_LIQUIDITY_FLOOR_QUANTILE": (
            float(raw)
            if (raw := backtest_cfg.get("liquidity_floor_quantile")) is not None
            else None
        ),
        "BACKTEST_WEIGHTING_LIQUIDITY_COL": str(
            backtest_cfg.get("weighting_liquidity_col", "medadv20_amount")
        ),
        "BACKTEST_MAX_TURNOVER_PER_REBALANCE": (
            float(raw)
            if (raw := backtest_cfg.get("max_turnover_per_rebalance")) is not None
            else None
        ),
        "BACKTEST_SELECTION_TIEBREAK_COL": _normalize_optional_text(
            backtest_cfg.get("selection_tiebreak_col")
        ),
        "BACKTEST_SELECTION_SCORE_BUCKET_SIZE": selection_score_bucket_size,
        "BACKTEST_SELECTION_SCORE_MARGIN": selection_score_margin,
        "BACKTEST_SELECTION_SCORE_MARGIN_RANK_LIMIT": selection_score_margin_rank_limit,
        "BACKTEST_SIGNAL_DIRECTION_RAW": _resolve_backtest_signal_direction(backtest_cfg),
        "BACKTEST_SHORT_K": (
            int(raw) if (raw := backtest_cfg.get("short_k")) is not None else None
        ),
        "BACKTEST_EXIT_MODE": backtest_exit_mode,
        "BACKTEST_EXIT_HORIZON_DAYS": backtest_exit_horizon_days,
        "BACKTEST_EXIT_PRICE_POLICY": backtest_exit_price_policy,
        "BACKTEST_EXIT_FALLBACK_POLICY": backtest_exit_fallback_policy,
        "BACKTEST_TRADABLE_COL": _normalize_optional_text(
            backtest_cfg.get("tradable_col", "is_tradable")
        ),
        **_resolve_backtest_postprocess_settings(backtest_cfg),
    }
