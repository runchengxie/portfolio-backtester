from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any, cast

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

BACKTEST_PRICING_CONTRACT_NAME = "cstree.backtest_pricing"
BACKTEST_PRICING_SCHEMA_VERSION = 1
BACKTEST_PRICING_KEY_COLUMNS = ("trade_date", "symbol")
DEFAULT_TRADABLE_FLAG_COLUMNS = ("is_tradable", "is_buy_tradable", "is_sell_tradable")

STRATEGY_SPEC_CONTRACT_NAME = "cstree.strategy_spec"
STRATEGY_SPEC_SCHEMA_VERSION = 1
STRATEGY_SPEC_REQUIRED_FIELDS = (
    "name",
    "type",
    "score_col",
    "top_k",
    "buffer_exit",
    "buffer_entry",
    "weighting",
    "long_only",
)

POSITIONS_BY_REBALANCE_CONTRACT_NAME = "cstree.positions_by_rebalance"
POSITIONS_BY_REBALANCE_SCHEMA_VERSION = 1
CANONICAL_POSITIONS_BY_REBALANCE_FILE = "positions_by_rebalance.csv"
POSITIONS_BY_REBALANCE_REQUIRED_COLUMNS = ("rebalance_date", "symbol", "weight")
POSITIONS_BY_REBALANCE_COMMON_COLUMNS = ("entry_date", "side", "signal", "rank")


@dataclass(frozen=True)
class BacktestPricingFrameContract:
    """Stable pricing input frame contract for backtest execution."""

    name: str = BACKTEST_PRICING_CONTRACT_NAME
    schema_version: int = BACKTEST_PRICING_SCHEMA_VERSION
    key_columns: tuple[str, ...] = BACKTEST_PRICING_KEY_COLUMNS


@dataclass(frozen=True)
class StrategySpecContract:
    """Stable portfolio construction strategy specification contract."""

    name: str = STRATEGY_SPEC_CONTRACT_NAME
    schema_version: int = STRATEGY_SPEC_SCHEMA_VERSION
    required_fields: tuple[str, ...] = STRATEGY_SPEC_REQUIRED_FIELDS


@dataclass(frozen=True)
class PositionsByRebalanceFrameContract:
    """Stable portfolio-construction positions artifact contract."""

    name: str = POSITIONS_BY_REBALANCE_CONTRACT_NAME
    schema_version: int = POSITIONS_BY_REBALANCE_SCHEMA_VERSION
    file_name: str = CANONICAL_POSITIONS_BY_REBALANCE_FILE
    required_columns: tuple[str, ...] = POSITIONS_BY_REBALANCE_REQUIRED_COLUMNS
    common_columns: tuple[str, ...] = POSITIONS_BY_REBALANCE_COMMON_COLUMNS
    date_column: str = "rebalance_date"
    symbol_column: str = "symbol"
    weight_column: str = "weight"


BACKTEST_PRICING_CONTRACT = BacktestPricingFrameContract()
STRATEGY_SPEC_CONTRACT = StrategySpecContract()
POSITIONS_BY_REBALANCE_CONTRACT = PositionsByRebalanceFrameContract()


@dataclass(frozen=True)
class GroupCap:
    column: str
    max_names: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StrategySpec:
    name: str
    type: str
    score_col: str
    top_k: int
    buffer_exit: int = 0
    buffer_entry: int = 0
    weighting: str = "equal"
    long_only: bool = True
    short_k: int | None = None
    group_cap: GroupCap | None = None
    execution: dict[str, Any] | None = None
    source: str = "explicit"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["group_cap"] = self.group_cap.to_dict() if self.group_cap else None
        return payload


def _dedupe(values: Iterable[str | None]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value is None:
            continue
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return tuple(ordered)


def _column_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return cast(pd.Series, frame[column])


def _parse_contract_dates(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    compact = text.str.fullmatch(r"\d{8}")
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    if bool(compact.any()):
        parsed.loc[compact] = pd.to_datetime(text.loc[compact], format="%Y%m%d", errors="coerce")
    remaining = ~compact
    if bool(remaining.any()):
        parsed.loc[remaining] = pd.to_datetime(text.loc[remaining], errors="coerce")
    return cast(pd.Series, parsed.dt.normalize())


def required_backtest_pricing_columns(
    *,
    entry_price_col: str,
    exit_price_col: str,
    amount_columns: Iterable[str] = (),
    tradable_col: str | None = None,
) -> tuple[str, ...]:
    return _dedupe(
        (
            *BACKTEST_PRICING_KEY_COLUMNS,
            entry_price_col,
            exit_price_col,
            *amount_columns,
            tradable_col,
        )
    )


def validate_backtest_pricing_frame(
    pricing: pd.DataFrame,
    *,
    entry_price_col: str,
    exit_price_col: str,
    amount_columns: Iterable[str] = (),
    tradable_col: str | None = None,
    require_two_trade_dates: bool = False,
) -> list[str]:
    amount_column_tuple = _dedupe(amount_columns)
    required_columns = required_backtest_pricing_columns(
        entry_price_col=entry_price_col,
        exit_price_col=exit_price_col,
        amount_columns=amount_column_tuple,
        tradable_col=tradable_col,
    )
    issues: list[str] = []
    missing = [column for column in required_columns if column not in pricing.columns]
    if missing:
        issues.append("missing columns: " + ", ".join(missing))
        return issues
    if pricing.empty:
        return issues

    trade_dates = cast(
        pd.Series,
        pd.to_datetime(_column_series(pricing, "trade_date"), errors="coerce"),
    )
    if bool(trade_dates.isna().any()):
        issues.append("trade_date must be datetime-like")
    elif require_two_trade_dates and int(trade_dates.dt.normalize().nunique()) < 2:
        issues.append("trade_date must contain at least two dates")

    symbols = _column_series(pricing, "symbol").astype("string")
    if bool(symbols.isna().any()) or bool(symbols.str.strip().eq("").any()):
        issues.append("symbol must be non-empty")

    numeric_columns = _dedupe((entry_price_col, exit_price_col, *amount_column_tuple))
    for column in numeric_columns:
        if not is_numeric_dtype(_column_series(pricing, column)):
            issues.append(f"{column} must be numeric")
    if tradable_col is not None:
        issues.extend(validate_tradable_flags_frame(pricing, columns=(tradable_col,)))
    return issues


def assert_backtest_pricing_frame(
    pricing: pd.DataFrame,
    *,
    entry_price_col: str,
    exit_price_col: str,
    amount_columns: Iterable[str] = (),
    tradable_col: str | None = None,
    require_two_trade_dates: bool = False,
) -> None:
    issues = validate_backtest_pricing_frame(
        pricing,
        entry_price_col=entry_price_col,
        exit_price_col=exit_price_col,
        amount_columns=amount_columns,
        tradable_col=tradable_col,
        require_two_trade_dates=require_two_trade_dates,
    )
    if issues:
        raise ValueError("Invalid backtest pricing frame: " + "; ".join(issues))


def validate_tradable_flags_frame(
    frame: pd.DataFrame,
    *,
    columns: Iterable[str] = DEFAULT_TRADABLE_FLAG_COLUMNS,
) -> list[str]:
    flag_columns = _dedupe(columns)
    issues: list[str] = []
    missing = [column for column in flag_columns if column not in frame.columns]
    if missing:
        issues.append("missing tradable flag columns: " + ", ".join(missing))
        return issues

    for column in flag_columns:
        if not is_bool_dtype(_column_series(frame, column)):
            issues.append(f"{column} must be boolean typed")
    return issues


def _non_empty(value: str, *, field: str) -> str | None:
    if value.strip():
        return None
    return f"{field} must be non-empty"


def validate_strategy_spec(spec: StrategySpec) -> list[str]:
    issues: list[str] = []
    for field in ("name", "type", "score_col", "weighting", "source"):
        issue = _non_empty(getattr(spec, field), field=field)
        if issue is not None:
            issues.append(issue)

    if spec.top_k <= 0:
        issues.append("top_k must be > 0")
    if spec.buffer_exit < 0:
        issues.append("buffer_exit must be >= 0")
    if spec.buffer_entry < 0:
        issues.append("buffer_entry must be >= 0")
    if spec.short_k is not None and spec.short_k <= 0:
        issues.append("short_k must be > 0 when provided")
    if spec.group_cap is not None:
        group_column_issue = _non_empty(spec.group_cap.column, field="group_cap.column")
        if group_column_issue is not None:
            issues.append(group_column_issue)
        if spec.group_cap.max_names <= 0:
            issues.append("group_cap.max_names must be > 0")
    return issues


def assert_strategy_spec(spec: StrategySpec) -> None:
    issues = validate_strategy_spec(spec)
    if issues:
        raise ValueError("Invalid strategy spec: " + "; ".join(issues))


def validate_positions_by_rebalance_frame(positions: pd.DataFrame) -> list[str]:
    """Return contract violations for a positions_by_rebalance artifact frame."""

    issues: list[str] = []
    missing = [
        column
        for column in POSITIONS_BY_REBALANCE_REQUIRED_COLUMNS
        if column not in positions.columns
    ]
    if missing:
        issues.append("missing columns: " + ", ".join(missing))
        return issues
    if positions.empty:
        return issues

    rebalance_dates = _parse_contract_dates(_column_series(positions, "rebalance_date"))
    if bool(rebalance_dates.isna().any()):
        issues.append("rebalance_date must be date-like")

    symbols = _column_series(positions, "symbol").astype("string")
    if bool(symbols.isna().any()) or bool(symbols.str.strip().eq("").any()):
        issues.append("symbol must be non-empty")

    weights = pd.to_numeric(_column_series(positions, "weight"), errors="coerce")
    if bool(weights.isna().any()):
        issues.append("weight must be numeric")
    return issues


def assert_positions_by_rebalance_frame(positions: pd.DataFrame) -> None:
    issues = validate_positions_by_rebalance_frame(positions)
    if issues:
        raise ValueError("Invalid positions_by_rebalance frame: " + "; ".join(issues))


__all__ = [
    "BACKTEST_PRICING_CONTRACT",
    "BACKTEST_PRICING_CONTRACT_NAME",
    "BACKTEST_PRICING_KEY_COLUMNS",
    "BACKTEST_PRICING_SCHEMA_VERSION",
    "CANONICAL_POSITIONS_BY_REBALANCE_FILE",
    "DEFAULT_TRADABLE_FLAG_COLUMNS",
    "POSITIONS_BY_REBALANCE_COMMON_COLUMNS",
    "POSITIONS_BY_REBALANCE_CONTRACT",
    "POSITIONS_BY_REBALANCE_CONTRACT_NAME",
    "POSITIONS_BY_REBALANCE_REQUIRED_COLUMNS",
    "POSITIONS_BY_REBALANCE_SCHEMA_VERSION",
    "STRATEGY_SPEC_CONTRACT",
    "STRATEGY_SPEC_CONTRACT_NAME",
    "STRATEGY_SPEC_REQUIRED_FIELDS",
    "STRATEGY_SPEC_SCHEMA_VERSION",
    "BacktestPricingFrameContract",
    "GroupCap",
    "PositionsByRebalanceFrameContract",
    "StrategySpec",
    "StrategySpecContract",
    "assert_backtest_pricing_frame",
    "assert_positions_by_rebalance_frame",
    "assert_strategy_spec",
    "required_backtest_pricing_columns",
    "validate_backtest_pricing_frame",
    "validate_positions_by_rebalance_frame",
    "validate_strategy_spec",
    "validate_tradable_flags_frame",
]
