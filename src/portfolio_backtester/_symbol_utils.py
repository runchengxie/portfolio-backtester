"""Inline symbol/path utilities formerly from market-data-platform.

Keeping these here makes portfolio-backtester self-contained so external
users can ``pip install`` and run without cloning market-data-platform.
"""

import os
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# canonicalize_symbol_columns
# ---------------------------------------------------------------------------

SYMBOL_COL = "symbol"
LEGACY_SYMBOL_COLUMNS = ("ts_code", "stock_ticker")
DEFAULT_SYMBOL_PRIORITY = ("symbol", "ts_code", "stock_ticker", "order_book_id")


def _column_series(df: pd.DataFrame, column: str) -> pd.Series:
    values = df.loc[:, column]
    if isinstance(values, pd.DataFrame):
        values = values.iloc[:, 0]
    return values


def _clean_symbol_series(values: pd.Series) -> pd.Series:
    series = values if isinstance(values, pd.Series) else pd.Series([values])
    return series.where(series.notna(), "").astype(str).str.strip()


def _resolve_symbol_series(
    df: pd.DataFrame,
    *,
    context: str,
    priority: Sequence[str] = DEFAULT_SYMBOL_PRIORITY,
) -> pd.Series:
    present_columns = [column for column in priority if column in df.columns]
    if not present_columns:
        raise SystemExit(f"{context} is missing symbol/stock_ticker/ts_code/order_book_id.")
    merged = _clean_symbol_series(_column_series(df, present_columns[0]))
    for column in present_columns[1:]:
        series = _clean_symbol_series(_column_series(df, column))
        merged = merged.where(merged != "", series)
    return merged


def _ensure_symbol_columns(
    df: pd.DataFrame,
    *,
    context: str,
    priority: Sequence[str] = DEFAULT_SYMBOL_PRIORITY,
) -> pd.DataFrame:
    normalized = df.copy()
    merged = _resolve_symbol_series(normalized, context=context, priority=priority)
    normalized[SYMBOL_COL] = merged
    return normalized


def _drop_legacy_symbol_columns(
    df: pd.DataFrame,
    *,
    drop_order_book_id: bool = False,
) -> pd.DataFrame:
    drop_columns = [*LEGACY_SYMBOL_COLUMNS]
    if drop_order_book_id:
        drop_columns.append("order_book_id")
    out = df.drop(columns=drop_columns, errors="ignore")
    out.attrs = dict(getattr(df, "attrs", {}))
    return out


def canonicalize_symbol_columns(
    df: pd.DataFrame,
    *,
    context: str,
    priority: Sequence[str] = DEFAULT_SYMBOL_PRIORITY,
    drop_order_book_id: bool = False,
) -> pd.DataFrame:
    """Ensure *df* has a canonical ``symbol`` column and drop legacy aliases.

    Scans the DataFrame for ``symbol``, ``ts_code``, ``stock_ticker``, or
    ``order_book_id`` (in that order by default) and merges the first
    non-empty value into a single ``symbol`` column.  Legacy columns
    (``ts_code``, ``stock_ticker``) are then dropped.
    """
    normalized = _ensure_symbol_columns(df, context=context, priority=priority)
    return _drop_legacy_symbol_columns(normalized, drop_order_book_id=drop_order_book_id)


# ---------------------------------------------------------------------------
# resolve_data_input_path
# ---------------------------------------------------------------------------

ENV_DATA_PLATFORM_ROOT = "DATA_PLATFORM_ROOT"
ENV_HK_DATA_PLATFORM_ROOT = "HK_DATA_PLATFORM_ROOT"
DATA_PLATFORM_PATH_PREFIXES = {
    ("artifacts", "assets"),
    ("artifacts", "metadata"),
    ("artifacts", "standardized"),
}


def _normalize_path_text(value: str | Path | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _resolve_env_path(env_name: str) -> str | None:
    return _normalize_path_text(os.getenv(env_name))


def _resolve_repo_path(path_text: str | Path) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path.cwd() / path).resolve()


def _resolve_artifacts_root(path_text: str | Path | None = None) -> Path:
    configured = (
        _normalize_path_text(path_text)
        or _resolve_env_path(ENV_DATA_PLATFORM_ROOT)
        or _resolve_env_path("CSTREE_ARTIFACTS_ROOT")
    )
    return _resolve_repo_path(configured or "artifacts")


def _resolve_hk_data_platform_root(path_text: str | Path | None = None) -> Path | None:
    configured = (
        _normalize_path_text(path_text)
        or _resolve_env_path(ENV_DATA_PLATFORM_ROOT)
        or _resolve_env_path(ENV_HK_DATA_PLATFORM_ROOT)
    )
    return _resolve_repo_path(configured) if configured is not None else None


def _data_platform_relative_path(path: Path) -> Path | None:
    if path.is_absolute():
        return None
    parts = path.parts
    if len(parts) < 2:
        return None
    if (parts[0], parts[1]) not in DATA_PLATFORM_PATH_PREFIXES:
        return None
    return Path(*parts[1:])


def _configured_data_input_path(path_text: str | Path) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path.absolute()
    data_root = _resolve_hk_data_platform_root()
    relative = _data_platform_relative_path(path)
    if data_root is not None and relative is not None:
        return (data_root / relative).absolute()
    return (Path.cwd() / path).absolute()


def resolve_data_input_path(path_text: str | Path) -> Path:
    """Resolve a data-input path to an absolute ``Path``.

    Absolute paths are returned unchanged.  Relative paths prefixed with
    ``artifacts/assets``, ``artifacts/metadata``, or
    ``artifacts/standardized`` are resolved against
    ``HK_DATA_PLATFORM_ROOT`` (or ``DATA_PLATFORM_ROOT``) when that
    environment variable is set.  All other relative paths are resolved
    relative to the current working directory.
    """
    return _configured_data_input_path(path_text).resolve()
