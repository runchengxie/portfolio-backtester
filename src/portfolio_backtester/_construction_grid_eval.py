from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pandas as pd

from ._construction_grid_io import (
    _coerce_bool,
    _first_non_empty,
    _get_nested,
    _periods_per_year,
    _read_returns_file,
    _resolve_path,
)
from ._construction_grid_signal import (
    _build_base_context,
    _prepare_signal_column,
)
from ._execution_models import ExitFallbackPolicy, ExitPricePolicy
from .benchmarking import build_benchmark_series
from .execution import build_execution_model
from .metrics import (
    daily_ic_series,
    estimate_turnover,
    quantile_returns,
    summarize_active_returns,
    summarize_ic,
)


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
            default_exit_price_policy=cast(ExitPricePolicy, exit_price_policy),
            default_exit_fallback_policy=cast(ExitFallbackPolicy, exit_fallback_policy),
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
    backtest_topk_fn: Any,
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
    backtest_topk_fn: Any,
    dynamic_ensemble_fn: Any | None,
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


def _resolve_backtest_topk_fn(candidate: Any) -> Any:
    if candidate is None:
        raise SystemExit(
            "Construction grid requires an injected backtest_topk_fn. "
            "Use the strategy CLI provided by strategy-pipeline or pass "
            "portfolio_backtester.engine.backtest_topk explicitly."
        )
    if not callable(candidate):
        raise SystemExit("Construction grid backtest_topk_fn must be callable.")
    return candidate


def build_construction_grid(
    config: dict[str, Any],
    *,
    config_dir: Path,
    backtest_topk_fn: Any | None = None,
    dynamic_ensemble_fn: Any | None = None,
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
