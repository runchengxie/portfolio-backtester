from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import pandas as pd

from .execution_sim.results import UnifiedLedger

BACKTEST_BUNDLE_SCHEMA_VERSION = "portfolio_backtester.backtest_result.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class BacktestEvidenceTier(StrEnum):
    DIAGNOSTIC = "diagnostic"
    EXECUTION_AWARE = "execution_aware"


_EXECUTION_CLOCK_FIELDS = (
    "schema_version",
    "timezone",
    "information_cutoff_at",
    "signal_at",
    "decision_at",
    "earliest_order_at",
    "execution_window_start_at",
    "execution_window_end_at",
    "valuation_at",
    "timing_policy_id",
    "trading_calendar_ref",
)

EXECUTION_AWARE_BUNDLE_FILES = frozenset(
    {
        "targets.parquet",
        "orders.parquet",
        "fills.parquet",
        "daily_positions.parquet",
        "daily_cash.parquet",
        "daily_nav.parquet",
        "cost_breakdown.parquet",
        "turnover_breakdown.parquet",
        "diagnostics.json",
    }
)


def _required_text(value: object, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} is required")
    return text


def _sha256(value: object, field: str) -> str:
    text = _required_text(value, field)
    if _SHA256.fullmatch(text) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return text


def _require_columns(frame: pd.DataFrame, columns: set[str], *, label: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: " + ", ".join(missing))


def _require_ledger_frames(ledger: UnifiedLedger) -> None:
    for name in (
        "targets",
        "orders",
        "fills",
        "daily_positions",
        "daily_cash",
        "daily_nav",
        "cost_breakdown",
        "turnover_breakdown",
    ):
        if not isinstance(getattr(ledger, name, None), pd.DataFrame):
            raise TypeError(f"UnifiedLedger.{name} must be a pandas DataFrame")


def reconcile_unified_ledger(
    ledger: UnifiedLedger,
    *,
    tolerance: float = 1e-8,
) -> dict[str, float | int | str]:
    _require_ledger_frames(ledger)
    _require_columns(
        ledger.daily_positions,
        {"trade_date", "positions_value"},
        label="daily_positions",
    )
    _require_columns(ledger.daily_cash, {"trade_date", "cash"}, label="daily_cash")
    _require_columns(ledger.daily_nav, {"trade_date", "nav"}, label="daily_nav")
    if ledger.daily_nav.empty:
        raise ValueError("daily_nav must contain at least one row for reconciliation")

    positions = ledger.daily_positions[["trade_date", "positions_value"]].copy()
    cash = ledger.daily_cash[["trade_date", "cash"]].copy()
    nav = ledger.daily_nav[["trade_date", "nav"]].copy()
    for frame in (positions, cash, nav):
        frame["trade_date"] = frame["trade_date"].astype(str)
        if frame["trade_date"].duplicated().any():
            raise ValueError("daily ledger trade_date values must be unique")

    merged = nav.merge(cash, on="trade_date", how="outer", validate="one_to_one").merge(
        positions,
        on="trade_date",
        how="outer",
        validate="one_to_one",
    )
    if merged[["nav", "cash", "positions_value"]].isna().any().any():
        raise ValueError("daily ledger date coverage must match across nav, cash and positions")

    for column in ("nav", "cash", "positions_value"):
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
    if merged[["nav", "cash", "positions_value"]].isna().any().any():
        raise ValueError("daily ledger accounting columns must be numeric")

    expected = merged["cash"] + merged["positions_value"]
    errors = (merged["nav"] - expected).abs()
    scale = pd.concat([merged["nav"].abs(), expected.abs()], axis=1).max(axis=1).clip(lower=1.0)
    if (errors > float(tolerance) * scale).any():
        raise ValueError("daily ledger must satisfy nav = cash + positions_value")
    return {
        "status": "passed",
        "rows": len(merged),
        "max_abs_error": float(errors.max()),
        "tolerance": float(tolerance),
    }


def _validate_execution_metadata(
    *,
    research_clock: Mapping[str, Any],
    backend_capabilities: Mapping[str, Any],
) -> None:
    for capability in ("order_lifecycle", "daily_ledger"):
        if backend_capabilities.get(capability) is not True:
            raise ValueError(f"execution-aware bundle requires backend capability {capability}")
    for field in _EXECUTION_CLOCK_FIELDS:
        value = research_clock.get(field)
        if value is None or (isinstance(value, str) and not value.strip()):
            raise ValueError(f"execution-aware bundle requires research_clock.{field}")
    if research_clock.get("schema_version") != "research.clock.v1":
        raise ValueError("execution-aware bundle requires research.clock.v1")


def validate_execution_aware_bundle_inputs(
    ledger: UnifiedLedger,
    *,
    research_clock: Mapping[str, Any],
    backend_capabilities: Mapping[str, Any],
) -> dict[str, float | int | str]:
    _require_ledger_frames(ledger)
    _validate_execution_metadata(
        research_clock=research_clock,
        backend_capabilities=backend_capabilities,
    )
    if not ledger.orders.empty:
        _require_columns(ledger.orders, {"order_id", "status"}, label="orders")
    if not ledger.fills.empty:
        _require_columns(ledger.fills, {"fill_id", "order_id"}, label="fills")
    return reconcile_unified_ledger(ledger)


@dataclass(frozen=True)
class BacktestBundleInventoryItem:
    path: str
    sha256: str
    required: bool
    rows: int | None = None

    def __post_init__(self) -> None:
        path = Path(_required_text(self.path, "inventory.path"))
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("inventory.path must be a safe relative path")
        _sha256(self.sha256, "inventory.sha256")
        if not isinstance(self.required, bool):
            raise TypeError("inventory.required must be a bool")
        if self.rows is not None and self.rows < 0:
            raise ValueError("inventory.rows must be non-negative")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> BacktestBundleInventoryItem:
        required = payload.get("required")
        if not isinstance(required, bool):
            raise TypeError("inventory.required must be a bool")
        raw_rows = payload.get("rows")
        rows = None if raw_rows is None else int(raw_rows)
        return cls(
            path=_required_text(payload.get("path"), "inventory.path"),
            sha256=_sha256(payload.get("sha256"), "inventory.sha256"),
            required=required,
            rows=rows,
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "required": self.required,
            "rows": self.rows,
        }


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _mapping_list(value: object, field: str) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    if not all(isinstance(item, Mapping) for item in value):
        raise ValueError(f"each {field} item must be an object")
    return tuple(dict(item) for item in value)


@dataclass(frozen=True)
class BacktestBundleManifest:
    run_id: str
    evidence_tier: BacktestEvidenceTier
    artifact_envelope: Mapping[str, Any]
    research_clock: Mapping[str, Any]
    backend: Mapping[str, Any]
    backend_capabilities: Mapping[str, Any]
    input_refs: tuple[Mapping[str, Any], ...]
    inventory: tuple[BacktestBundleInventoryItem, ...]
    reconciliation: Mapping[str, Any]
    schema_version: str = BACKTEST_BUNDLE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != BACKTEST_BUNDLE_SCHEMA_VERSION:
            raise ValueError(f"unsupported backtest bundle schema {self.schema_version!r}")
        _required_text(self.run_id, "run_id")
        _required_text(self.backend.get("name"), "backend.name")
        if self.artifact_envelope.get("schema_version") != "research.artifact-envelope.v2":
            raise ValueError("artifact_envelope must use research.artifact-envelope.v2")
        _sha256(self.artifact_envelope.get("content_sha256"), "artifact_envelope.content_sha256")
        paths = [item.path for item in self.inventory]
        if len(paths) != len(set(paths)):
            raise ValueError("inventory paths must be unique")
        if self.evidence_tier is BacktestEvidenceTier.EXECUTION_AWARE:
            _validate_execution_metadata(
                research_clock=self.research_clock,
                backend_capabilities=self.backend_capabilities,
            )
            missing = sorted(EXECUTION_AWARE_BUNDLE_FILES - set(paths))
            if missing:
                raise ValueError(
                    "missing required execution-aware bundle files: " + ", ".join(missing)
                )
            required_by_path = {item.path: item.required for item in self.inventory}
            not_required = sorted(
                path for path in EXECUTION_AWARE_BUNDLE_FILES if not required_by_path[path]
            )
            if not_required:
                raise ValueError(
                    "execution-aware bundle files must be marked required: "
                    + ", ".join(not_required)
                )
            if self.reconciliation.get("status") != "passed":
                raise ValueError("execution-aware bundle requires passed reconciliation")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> BacktestBundleManifest:
        schema_version = _required_text(payload.get("schema_version"), "schema_version")
        if schema_version != BACKTEST_BUNDLE_SCHEMA_VERSION:
            raise ValueError(f"unsupported backtest bundle schema {schema_version!r}")
        raw_inventory = payload.get("inventory")
        if not isinstance(raw_inventory, list):
            raise ValueError("inventory must be a list")
        if not all(isinstance(item, Mapping) for item in raw_inventory):
            raise ValueError("each inventory item must be an object")
        return cls(
            schema_version=schema_version,
            run_id=_required_text(payload.get("run_id"), "run_id"),
            evidence_tier=BacktestEvidenceTier(
                _required_text(payload.get("evidence_tier"), "evidence_tier")
            ),
            artifact_envelope=dict(_mapping(payload.get("artifact_envelope"), "artifact_envelope")),
            research_clock=dict(_mapping(payload.get("research_clock"), "research_clock")),
            backend=dict(_mapping(payload.get("backend"), "backend")),
            backend_capabilities=dict(
                _mapping(payload.get("backend_capabilities"), "backend_capabilities")
            ),
            input_refs=_mapping_list(payload.get("input_refs"), "input_refs"),
            inventory=tuple(
                BacktestBundleInventoryItem.from_mapping(item) for item in raw_inventory
            ),
            reconciliation=dict(_mapping(payload.get("reconciliation"), "reconciliation")),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "evidence_tier": self.evidence_tier.value,
            "artifact_envelope": dict(self.artifact_envelope),
            "research_clock": dict(self.research_clock),
            "backend": dict(self.backend),
            "backend_capabilities": dict(self.backend_capabilities),
            "input_refs": [dict(item) for item in self.input_refs],
            "inventory": [item.to_mapping() for item in self.inventory],
            "reconciliation": dict(self.reconciliation),
        }


__all__ = [
    "BACKTEST_BUNDLE_SCHEMA_VERSION",
    "EXECUTION_AWARE_BUNDLE_FILES",
    "BacktestBundleInventoryItem",
    "BacktestBundleManifest",
    "BacktestEvidenceTier",
    "reconcile_unified_ledger",
    "validate_execution_aware_bundle_inputs",
]
