"""Contracts for portfolio backtest output frames."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import cast

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

TRADABLE_FLAGS_CONTRACT_NAME = "portfolio_backtester.tradable_flags"
TRADABLE_FLAGS_SCHEMA_VERSION = 1
DEFAULT_TRADABLE_FLAG_COLUMNS = ("is_tradable", "is_buy_tradable", "is_sell_tradable")
BACKTEST_RETURN_CONTRACT_NAME = "portfolio_backtester.backtest_return_series"
BACKTEST_RETURN_SCHEMA_VERSION = 1
BACKTEST_RETURN_PERIOD_COLUMN = "period_end"
BACKTEST_RETURN_VALUE_COLUMNS = ("net_return", "gross_return", "turnover")
BACKTEST_PERIODS_CONTRACT_NAME = "portfolio_backtester.backtest_periods"
BACKTEST_PERIODS_SCHEMA_VERSION = 1
BACKTEST_PERIOD_COLUMNS = (
    "rebalance_date",
    "entry_idx",
    "planned_exit_idx",
    "exit_idx",
    "entry_date",
    "planned_exit_date",
    "exit_date",
    "exit_delay_steps",
)
BACKTEST_PERIOD_DATE_COLUMNS = (
    "rebalance_date",
    "entry_date",
    "planned_exit_date",
    "exit_date",
)
BACKTEST_PERIOD_INDEX_COLUMNS = (
    "entry_idx",
    "planned_exit_idx",
    "exit_idx",
    "exit_delay_steps",
)


@dataclass(frozen=True)
class TradableFlagsContract:
    name: str = TRADABLE_FLAGS_CONTRACT_NAME
    schema_version: int = TRADABLE_FLAGS_SCHEMA_VERSION
    default_columns: tuple[str, ...] = DEFAULT_TRADABLE_FLAG_COLUMNS


@dataclass(frozen=True)
class BacktestReturnSeriesContract:
    name: str = BACKTEST_RETURN_CONTRACT_NAME
    schema_version: int = BACKTEST_RETURN_SCHEMA_VERSION
    period_column: str = BACKTEST_RETURN_PERIOD_COLUMN
    value_columns: tuple[str, ...] = BACKTEST_RETURN_VALUE_COLUMNS


@dataclass(frozen=True)
class BacktestPeriodsContract:
    name: str = BACKTEST_PERIODS_CONTRACT_NAME
    schema_version: int = BACKTEST_PERIODS_SCHEMA_VERSION
    required_columns: tuple[str, ...] = BACKTEST_PERIOD_COLUMNS
    date_columns: tuple[str, ...] = BACKTEST_PERIOD_DATE_COLUMNS
    index_columns: tuple[str, ...] = BACKTEST_PERIOD_INDEX_COLUMNS


TRADABLE_FLAGS_CONTRACT = TradableFlagsContract()
BACKTEST_RETURN_CONTRACT = BacktestReturnSeriesContract()
BACKTEST_PERIODS_CONTRACT = BacktestPeriodsContract()


def _column_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return cast(pd.Series, frame[column])


def _dedupe(values: Iterable[str | None]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value.strip() for value in values if value and value.strip()))


def validate_tradable_flags_frame(
    frame: pd.DataFrame,
    *,
    columns: Iterable[str] = DEFAULT_TRADABLE_FLAG_COLUMNS,
) -> list[str]:
    flag_columns = _dedupe(columns)
    missing = [column for column in flag_columns if column not in frame.columns]
    if missing:
        return ["missing tradable flag columns: " + ", ".join(missing)]
    return [
        f"{column} must be boolean typed"
        for column in flag_columns
        if not is_bool_dtype(_column_series(frame, column))
    ]


def assert_tradable_flags_frame(
    frame: pd.DataFrame,
    *,
    columns: Iterable[str] = DEFAULT_TRADABLE_FLAG_COLUMNS,
) -> None:
    issues = validate_tradable_flags_frame(frame, columns=columns)
    if issues:
        raise ValueError("Invalid tradable flags frame: " + "; ".join(issues))


def build_backtest_return_frame(series: pd.Series, *, value_column: str) -> pd.DataFrame:
    if series is None or series.empty:
        return pd.DataFrame(columns=pd.Index((BACKTEST_RETURN_PERIOD_COLUMN, value_column)))
    return pd.DataFrame(
        {
            BACKTEST_RETURN_PERIOD_COLUMN: pd.to_datetime(series.index, errors="coerce"),
            value_column: pd.to_numeric(series, errors="coerce").to_numpy(),
        }
    )


def validate_backtest_return_frame(frame: pd.DataFrame, *, value_column: str) -> list[str]:
    required = (BACKTEST_RETURN_PERIOD_COLUMN, value_column)
    missing = [column for column in required if column not in frame]
    if missing:
        return ["missing columns: " + ", ".join(missing)]
    if frame.empty:
        return []
    issues: list[str] = []
    if (
        pd.to_datetime(_column_series(frame, BACKTEST_RETURN_PERIOD_COLUMN), errors="coerce")
        .isna()
        .any()
    ):
        issues.append(f"{BACKTEST_RETURN_PERIOD_COLUMN} must be datetime-like")
    if not is_numeric_dtype(_column_series(frame, value_column)):
        issues.append(f"{value_column} must be numeric")
    return issues


def assert_backtest_return_frame(frame: pd.DataFrame, *, value_column: str) -> None:
    issues = validate_backtest_return_frame(frame, value_column=value_column)
    if issues:
        raise ValueError("Invalid backtest return frame: " + "; ".join(issues))


def build_backtest_periods_frame(period_info: Iterable[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(list(period_info), columns=pd.Index(BACKTEST_PERIOD_COLUMNS))
    for column in BACKTEST_PERIOD_DATE_COLUMNS:
        frame[column] = pd.to_datetime(_column_series(frame, column), errors="coerce")
    for column in BACKTEST_PERIOD_INDEX_COLUMNS:
        frame[column] = pd.to_numeric(_column_series(frame, column), errors="coerce").astype(
            "Int64"
        )
    return frame


def validate_backtest_periods_frame(frame: pd.DataFrame) -> list[str]:
    missing = [column for column in BACKTEST_PERIOD_COLUMNS if column not in frame]
    if missing:
        return ["missing columns: " + ", ".join(missing)]
    if frame.empty:
        return []
    issues: list[str] = []
    for column in BACKTEST_PERIOD_DATE_COLUMNS:
        if pd.to_datetime(_column_series(frame, column), errors="coerce").isna().any():
            issues.append(f"{column} must be datetime-like")
    for column in BACKTEST_PERIOD_INDEX_COLUMNS:
        if not is_numeric_dtype(_column_series(frame, column)):
            issues.append(f"{column} must be numeric")
    return issues


def assert_backtest_periods_frame(frame: pd.DataFrame) -> None:
    issues = validate_backtest_periods_frame(frame)
    if issues:
        raise ValueError("Invalid backtest periods frame: " + "; ".join(issues))


__all__ = [
    "BACKTEST_PERIODS_CONTRACT",
    "BACKTEST_PERIODS_CONTRACT_NAME",
    "BACKTEST_PERIODS_SCHEMA_VERSION",
    "BACKTEST_PERIOD_COLUMNS",
    "BACKTEST_PERIOD_DATE_COLUMNS",
    "BACKTEST_PERIOD_INDEX_COLUMNS",
    "BACKTEST_RETURN_CONTRACT",
    "BACKTEST_RETURN_CONTRACT_NAME",
    "BACKTEST_RETURN_PERIOD_COLUMN",
    "BACKTEST_RETURN_SCHEMA_VERSION",
    "BACKTEST_RETURN_VALUE_COLUMNS",
    "DEFAULT_TRADABLE_FLAG_COLUMNS",
    "TRADABLE_FLAGS_CONTRACT",
    "TRADABLE_FLAGS_CONTRACT_NAME",
    "TRADABLE_FLAGS_SCHEMA_VERSION",
    "BacktestPeriodsContract",
    "BacktestReturnSeriesContract",
    "TradableFlagsContract",
    "assert_backtest_periods_frame",
    "assert_backtest_return_frame",
    "assert_tradable_flags_frame",
    "build_backtest_periods_frame",
    "build_backtest_return_frame",
    "validate_backtest_periods_frame",
    "validate_backtest_return_frame",
    "validate_tradable_flags_frame",
]
