from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from portfolio_backtester._symbol_utils import canonicalize_symbol_columns

from .rebalance import get_rebalance_dates


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
