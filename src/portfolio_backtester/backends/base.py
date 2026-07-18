"""Framework-neutral backtest backend protocol and canonical result contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from math import isfinite
from typing import Any, Protocol, TypeVar, runtime_checkable

import numpy as np
import pandas as pd

RequestT = TypeVar("RequestT", contravariant=True)

CANONICAL_BACKTEST_RESULT_SCHEMA = "canonical_backtest_result.v1"


@dataclass(frozen=True)
class BackendCapabilities:
    """Features a backend can truthfully provide."""

    target_generation: bool = False
    order_lifecycle: bool = False
    partial_fills: bool = False
    daily_ledger: bool = False
    long_short: bool = False
    market_rules: tuple[str, ...] = ()

    def to_mapping(self) -> dict[str, Any]:
        return {
            "target_generation": bool(self.target_generation),
            "order_lifecycle": bool(self.order_lifecycle),
            "partial_fills": bool(self.partial_fills),
            "daily_ledger": bool(self.daily_ledger),
            "long_short": bool(self.long_short),
            "market_rules": list(self.market_rules),
        }


@dataclass(frozen=True)
class CanonicalBacktestResult:
    """Stable result envelope shared by native and external backends.

    Unsupported tables remain empty and are declared through ``capabilities``.
    This prevents a period-return replay from pretending it produced actual
    broker orders or fills.
    """

    backend_name: str
    performance: pd.DataFrame
    positions: pd.DataFrame
    capabilities: BackendCapabilities
    orders: pd.DataFrame = field(default_factory=pd.DataFrame)
    fills: pd.DataFrame = field(default_factory=pd.DataFrame)
    daily_ledger: pd.DataFrame = field(default_factory=pd.DataFrame)
    summary: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = CANONICAL_BACKTEST_RESULT_SCHEMA

    def validate(self) -> None:
        if not self.backend_name.strip():
            raise ValueError("CanonicalBacktestResult.backend_name must be non-empty.")
        if self.schema_version != CANONICAL_BACKTEST_RESULT_SCHEMA:
            raise ValueError(f"Unsupported canonical result schema: {self.schema_version}")
        for name, frame in self.frames().items():
            if not isinstance(frame, pd.DataFrame):
                raise TypeError(f"CanonicalBacktestResult.{name} must be a pandas DataFrame.")

        _require_columns(self.performance, {"period_end"}, label="performance")
        if not self.positions.empty:
            _require_columns(
                self.positions,
                {"rebalance_date", "symbol", "weight"},
                label="positions",
            )
        if self.capabilities.order_lifecycle and not self.orders.empty:
            _require_columns(self.orders, {"order_id", "status"}, label="orders")
        elif not self.capabilities.order_lifecycle and (
            not self.orders.empty or not self.fills.empty
        ):
            raise ValueError(
                "A backend without order_lifecycle capability must not emit orders or fills."
            )
        if not self.orders.empty:
            _assert_unique(self.orders, "order_id", label="orders")
        if not self.fills.empty:
            _require_columns(self.fills, {"fill_id", "order_id"}, label="fills")
            _assert_unique(self.fills, "fill_id", label="fills")
            unknown = set(self.fills["order_id"].astype(str)) - set(
                self.orders["order_id"].astype(str)
            )
            if unknown:
                raise ValueError(
                    "Fills reference unknown order_id values: "
                    + ", ".join(sorted(unknown))
                )
        if self.capabilities.daily_ledger and not self.daily_ledger.empty:
            _require_columns(
                self.daily_ledger,
                {"trade_date", "cash", "positions_value", "nav"},
                label="daily_ledger",
            )
            _assert_daily_ledger_balanced(self.daily_ledger)
        elif not self.capabilities.daily_ledger and not self.daily_ledger.empty:
            raise ValueError(
                "A backend without daily_ledger capability must not emit daily ledger rows."
            )

        _assert_json_compatible(self.capabilities.to_mapping(), label="capabilities")
        _assert_json_compatible(self.summary, label="summary")
        _assert_json_compatible(self.metadata, label="metadata")

    def frames(self) -> dict[str, pd.DataFrame]:
        return {
            "performance": self.performance,
            "positions": self.positions,
            "orders": self.orders,
            "fills": self.fills,
            "daily_ledger": self.daily_ledger,
        }

    def describe(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "backend_name": self.backend_name,
            "capabilities": self.capabilities.to_mapping(),
            "row_counts": {name: int(frame.shape[0]) for name, frame in self.frames().items()},
            "summary": to_json_compatible(self.summary),
            "metadata": to_json_compatible(self.metadata),
        }


@runtime_checkable
class BacktestBackend(Protocol[RequestT]):
    """Minimal backend boundary; adapters own translation, never public contracts."""

    name: str
    capabilities: BackendCapabilities

    def run(self, request: RequestT) -> CanonicalBacktestResult: ...


class BackendRegistry:
    """Explicit backend registry with no import-time plugin discovery."""

    def __init__(self) -> None:
        self._backends: dict[str, BacktestBackend[Any]] = {}

    def register(self, backend: BacktestBackend[Any]) -> None:
        name = str(backend.name).strip()
        if not name:
            raise ValueError("Backtest backend name must be non-empty.")
        if name in self._backends:
            raise ValueError(f"Backtest backend already registered: {name}")
        self._backends[name] = backend

    def get(self, name: str) -> BacktestBackend[Any]:
        try:
            return self._backends[name]
        except KeyError as exc:
            known = ", ".join(sorted(self._backends)) or "<none>"
            raise KeyError(f"Unknown backtest backend {name!r}; registered: {known}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._backends))

    def run(self, name: str, request: Any) -> CanonicalBacktestResult:
        result = self.get(name).run(request)
        result.validate()
        return result


def to_json_compatible(value: Any) -> Any:
    """Normalize common numeric and timestamp scalars into strict JSON values."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if value is pd.NA:
        return None
    if isinstance(value, Mapping):
        return {str(key): to_json_compatible(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [to_json_compatible(item) for item in value]
    raise TypeError(f"Unsupported value in framework-neutral metadata: {type(value).__name__}")


def _assert_json_compatible(value: Any, *, label: str) -> None:
    try:
        to_json_compatible(value)
    except TypeError as exc:
        raise TypeError(f"CanonicalBacktestResult.{label} is not JSON-compatible") from exc


def _require_columns(frame: pd.DataFrame, columns: set[str], *, label: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"Canonical {label} frame is missing columns: " + ", ".join(missing))


def _assert_unique(frame: pd.DataFrame, column: str, *, label: str) -> None:
    values = frame[column].astype(str)
    if values.duplicated(keep=False).any():
        raise ValueError(f"Canonical {label}.{column} values must be unique.")


def _assert_daily_ledger_balanced(frame: pd.DataFrame, *, tolerance: float = 1e-8) -> None:
    cash = pd.to_numeric(frame["cash"], errors="coerce")
    positions = pd.to_numeric(frame["positions_value"], errors="coerce")
    nav = pd.to_numeric(frame["nav"], errors="coerce")
    if cash.isna().any() or positions.isna().any() or nav.isna().any():
        raise ValueError("Canonical daily_ledger accounting columns must be numeric.")
    expected = cash + positions
    scale = pd.concat([nav.abs(), expected.abs()], axis=1).max(axis=1).clip(lower=1.0)
    if ((nav - expected).abs() > tolerance * scale).any():
        raise ValueError("Canonical daily_ledger must satisfy nav = cash + positions_value.")


__all__ = [
    "CANONICAL_BACKTEST_RESULT_SCHEMA",
    "BackendCapabilities",
    "BackendRegistry",
    "BacktestBackend",
    "CanonicalBacktestResult",
    "to_json_compatible",
]
