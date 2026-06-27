from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .execution import DetailedTradeFeeModel, build_execution_model

GRID_FIELDNAMES = [
    "portfolio_value",
    "participation_rate",
    "passed",
    "binding_constraints",
    "ideal_total_return",
    "exec_total_return",
    "return_degradation",
    "return_retention",
    "ideal_sharpe",
    "exec_sharpe",
    "sharpe_degradation",
    "sharpe_retention",
    "ideal_max_drawdown",
    "exec_max_drawdown",
    "fill_ratio",
    "buy_fill_ratio",
    "sell_fill_ratio",
    "unfilled_notional",
    "avg_cash_weight",
    "avg_target_cash_weight",
    "avg_execution_shortfall_cash_weight",
    "final_cash_weight",
    "final_target_cash_weight",
    "final_execution_shortfall_cash_weight",
    "abandoned_buy_orders",
    "abandoned_buy_order_rate",
    "delayed_sell_orders",
    "delayed_sell_order_rate",
    "p95_participation",
    "p99_participation",
    "p95_capacity_utilization",
    "orders",
    "daily_rows",
    "status",
]


def resolve_path(path_text: str | Path | None, *, base_dir: Path | None = None) -> Path | None:
    if path_text is None or str(path_text).strip() == "":
        return None
    candidate = Path(path_text).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    if base_dir is not None:
        return (base_dir / candidate).resolve()
    return (Path.cwd() / candidate).resolve()


def read_json_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"Failed to read JSON file: {path} ({exc})") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"JSON file must contain an object: {path}")
    return payload


def read_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise SystemExit(f"Failed to read YAML file: {path} ({exc})") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"YAML file must contain an object: {path}")
    return payload


def capacity_cfg(config: Mapping[str, Any]) -> Mapping[str, Any]:
    value = config.get("capacity_report", {})
    return value if isinstance(value, Mapping) else {}


def mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def as_list(value: object) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, Iterable):
        return list(value)
    return [value]


def float_grid(
    *,
    cli_values: list[float] | None,
    cfg_values: object,
    fallback: tuple[float, ...],
    label: str,
) -> list[float]:
    values = list(cli_values or [])
    if not values:
        values = [float(item) for item in as_list(cfg_values)]
    if not values:
        values = list(fallback)
    out = sorted({float(item) for item in values if np.isfinite(float(item)) and float(item) > 0})
    if not out:
        raise SystemExit(f"{label} must include at least one positive value.")
    return out


def parse_csv_floats(values: list[str] | None) -> list[float] | None:
    if not values:
        return None
    parsed: list[float] = []
    for value in values:
        for item in str(value).split(","):
            text = item.strip()
            if text:
                parsed.append(float(text))
    return parsed


def artifact_path(run_dir: Path, value: object) -> Path | None:
    if not value:
        return None
    raw = Path(str(value)).expanduser()
    candidates = [raw] if raw.is_absolute() else [run_dir / raw, Path.cwd() / raw]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def summary_positions_path(summary: Mapping[str, Any], run_dir: Path) -> Path | None:
    positions = mapping(summary.get("positions"))
    path = artifact_path(run_dir, positions.get("by_rebalance_file"))
    if path is not None:
        return path
    strategy = mapping(positions.get("strategy"))
    return artifact_path(run_dir, strategy.get("positions_file"))


def resolve_positions_path(
    *,
    run_dir: Path,
    summary: Mapping[str, Any],
    args: argparse.Namespace,
    cfg: Mapping[str, Any],
) -> Path:
    explicit = resolve_path(args.positions_file or cfg.get("positions_file"), base_dir=run_dir)
    candidates = [
        explicit,
        summary_positions_path(summary, run_dir),
        run_dir / "positions_by_rebalance.csv",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate.resolve()
    raise SystemExit(
        "No positions_by_rebalance file found. Pass --positions-file or run from a complete run_dir."
    )


def summary_pricing_path(summary: Mapping[str, Any], run_dir: Path) -> Path | None:
    dataset = mapping(summary.get("dataset"))
    dataset_path = artifact_path(run_dir, dataset.get("file"))
    if dataset_path is not None and dataset_path.exists():
        return dataset_path
    signals = mapping(summary.get("signals"))
    scored_path = artifact_path(run_dir, signals.get("legacy_eval_scored_file"))
    if scored_path is not None and scored_path.exists():
        return scored_path
    return None


def resolve_pricing_path(
    *,
    run_dir: Path,
    summary: Mapping[str, Any],
    args: argparse.Namespace,
    cfg: Mapping[str, Any],
) -> Path:
    explicit = resolve_path(args.pricing_file or cfg.get("pricing_file"), base_dir=run_dir)
    candidates = [
        explicit,
        run_dir / "backtest_pricing.parquet",
        run_dir / "backtest_pricing.csv",
        summary_pricing_path(summary, run_dir),
        run_dir / "dataset.parquet",
        run_dir / "eval_scored.parquet",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate.resolve()
    raise SystemExit(
        "No pricing panel found. Pass --pricing-file with trade_date, symbol, price, and liquidity columns."
    )


def read_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Input file not found: {path}")
    suffix = path.suffix.lower()
    frame = pd.read_parquet(path) if suffix in {".parquet", ".pq"} else pd.read_csv(path)
    if "trade_date" not in frame.columns or "symbol" not in frame.columns:
        frame = frame.reset_index()
    return frame


def normalize_date_column(frame: pd.DataFrame, column: str) -> pd.Series:
    text = frame[column].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    parsed = pd.to_datetime(text, errors="coerce")
    mask = text.str.fullmatch(r"\d{8}")
    if mask.any():
        parsed.loc[mask] = pd.to_datetime(text.loc[mask], format="%Y%m%d", errors="coerce")
    return parsed.dt.normalize()


def normalize_symbol_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "symbol" not in out.columns:
        for candidate in ("stock_ticker", "ts_code", "order_book_id", "ticker"):
            if candidate in out.columns:
                out["symbol"] = out[candidate].astype(str)
                break
    if "symbol" not in out.columns:
        raise SystemExit(
            "Input frame must include symbol, stock_ticker, ts_code, or order_book_id."
        )
    out["symbol"] = out["symbol"].astype(str)
    return out


def normalize_pricing_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = normalize_symbol_columns(frame)
    if "trade_date" not in out.columns and "date" in out.columns:
        out["trade_date"] = out["date"]
    if "trade_date" not in out.columns:
        raise SystemExit("Pricing frame must include trade_date or date.")
    out["trade_date"] = normalize_date_column(out, "trade_date")
    out = out.dropna(subset=["trade_date", "symbol"]).copy()
    return out.drop_duplicates(subset=["trade_date", "symbol"]).sort_values(
        ["trade_date", "symbol"]
    )


def normalize_positions_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = normalize_symbol_columns(frame)
    for column in ("rebalance_date", "entry_date"):
        if column not in out.columns:
            raise SystemExit(f"Positions frame must include {column}.")
        out[column] = normalize_date_column(out, column)
    if "weight" not in out.columns:
        raise SystemExit("Positions frame must include weight.")
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce")
    out = out.dropna(subset=["rebalance_date", "entry_date", "symbol", "weight"]).copy()
    return out.sort_values(["rebalance_date", "entry_date", "symbol"])


def merge_execution_cfg(
    *,
    execution_cfg: Mapping[str, Any],
    backtest_cfg: Mapping[str, Any],
) -> Mapping[str, Any]:
    backtest_execution_cfg = backtest_cfg.get("execution")
    if not isinstance(backtest_execution_cfg, Mapping):
        return execution_cfg
    merged = dict(execution_cfg)
    for key, value in backtest_execution_cfg.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            nested = dict(merged[key])
            nested.update(value)
            merged[key] = nested
        else:
            merged[key] = value
    return merged


def build_execution_context(config: Mapping[str, Any]) -> dict[str, Any]:
    data_cfg = mapping(config.get("data"))
    backtest_cfg = mapping(config.get("backtest"))
    execution_cfg = mapping(config.get("execution"))
    price_col = str(data_cfg.get("price_col", "close")).strip() or "close"
    cost_bps = float(backtest_cfg.get("transaction_cost_bps", 0.0) or 0.0)
    exit_price_policy = str(backtest_cfg.get("exit_price_policy", "strict")).strip().lower()
    exit_fallback_policy = str(backtest_cfg.get("exit_fallback_policy", "ffill")).strip().lower()
    execution_model = build_execution_model(
        merge_execution_cfg(execution_cfg=execution_cfg, backtest_cfg=backtest_cfg),
        default_cost_bps=cost_bps,
        default_exit_price_policy=exit_price_policy,
        default_exit_fallback_policy=exit_fallback_policy,
        default_price_col=price_col,
    )
    return {
        "price_col": execution_model.entry_policy.price_col,
        "tradable_col": str(backtest_cfg.get("tradable_col", "is_tradable") or "").strip() or None,
        "transaction_cost_bps": cost_bps,
        "trading_days_per_year": int(backtest_cfg.get("trading_days_per_year", 252) or 252),
        "execution_model": execution_model,
        "trade_fee_model": execution_model.cost_model
        if isinstance(execution_model.cost_model, DetailedTradeFeeModel)
        else None,
        "default_liquidity_col": str(
            getattr(execution_model.slippage_model, "amount_col", "medadv20_amount")
            or "medadv20_amount"
        ),
    }


def execution_sim_raw(config: Mapping[str, Any]) -> dict[str, Any]:
    raw = mapping(mapping(config.get("backtest")).get("execution_sim"))
    return dict(raw)


def coerce_liquidity_cols(
    *,
    args: argparse.Namespace,
    cfg: Mapping[str, Any],
    sim_raw: Mapping[str, Any],
) -> list[str] | None:
    if args.liquidity_col:
        return [str(item).strip() for item in args.liquidity_col if str(item).strip()]
    cfg_cols = as_list(cfg.get("liquidity_cols") or cfg.get("liquidity_col"))
    if cfg_cols:
        return [str(item).strip() for item in cfg_cols if str(item).strip()]
    sim_cols = as_list(sim_raw.get("liquidity_cols") or sim_raw.get("liquidity_col"))
    if sim_cols:
        return [str(item).strip() for item in sim_cols if str(item).strip()]
    return None


def json_default(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if pd.isna(value):
        return None
    return str(value)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=GRID_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
