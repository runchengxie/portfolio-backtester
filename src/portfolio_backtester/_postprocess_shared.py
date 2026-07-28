"""Shared helpers for position post-processing (config checks, table loading)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from portfolio_backtester._symbol_utils import resolve_data_input_path


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
