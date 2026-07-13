"""Versioned JSON exchange for out-of-process LEAN golden reference runs."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ..backends import BacktestBackendResult, PositionReplayRequest

SCENARIO_SCHEMA = "lean_golden_scenario.v1"
RESULT_SCHEMA = "lean_golden_result.v1"


class LeanGoldenExchangeError(ValueError):
    """Raised when a LEAN golden envelope is invalid or has been modified."""


@dataclass(frozen=True)
class LeanGoldenScenario:
    """Framework-neutral inputs that a separate LEAN harness can consume."""

    scenario_id: str
    description: str
    positions: tuple[Mapping[str, Any], ...]
    pricing: tuple[Mapping[str, Any], ...]
    periods: tuple[Mapping[str, Any], ...]
    config: Mapping[str, Any]
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.scenario_id.strip() or not self.description.strip():
            raise ValueError("scenario_id and description must not be empty.")

    @classmethod
    def from_position_replay_request(
        cls,
        *,
        scenario_id: str,
        description: str,
        request: PositionReplayRequest,
        metadata: Mapping[str, Any] | None = None,
    ) -> LeanGoldenScenario:
        config = {
            "price_col": request.config.price_col,
            "entry_price_col": request.config.entry_price_col,
            "exit_price_col": request.config.exit_price_col,
            "transaction_cost_bps": request.config.transaction_cost_bps,
            "trading_days_per_year": request.config.trading_days_per_year,
            "long_only": request.config.long_only,
            "preserve_gross_exposure": request.config.preserve_gross_exposure,
            "exit_price_policy": request.config.exit_price_policy,
            "exit_fallback_policy": request.config.exit_fallback_policy,
            "tradable_col": request.config.tradable_col,
        }
        return cls(
            scenario_id=scenario_id,
            description=description,
            positions=_frame_records(request.positions),
            pricing=_frame_records(request.pricing),
            periods=_frame_records(request.periods),
            config=config,
            metadata=dict(metadata or {}),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "description": self.description,
            "positions": _sorted_records(self.positions),
            "pricing": _sorted_records(self.pricing),
            "periods": _sorted_records(self.periods),
            "config": _json_ready(self.config),
            "metadata": _json_ready(self.metadata),
        }


@dataclass(frozen=True)
class LeanGoldenResult:
    """Canonical performance, positions and fill evidence from a reference run."""

    scenario_sha256: str
    backend: str
    performance: tuple[Mapping[str, Any], ...]
    positions: tuple[Mapping[str, Any], ...]
    fills: tuple[Mapping[str, Any], ...]
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if len(self.scenario_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.scenario_sha256
        ):
            raise ValueError("scenario_sha256 must be a lowercase SHA-256 digest.")
        if not self.backend.strip():
            raise ValueError("backend must not be empty.")

    @classmethod
    def from_backend_result(
        cls,
        *,
        scenario_sha256: str,
        result: BacktestBackendResult,
        fills: Sequence[Mapping[str, Any]] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> LeanGoldenResult:
        result_metadata = {"source_backend_metadata": result.metadata, **dict(metadata or {})}
        return cls(
            scenario_sha256=scenario_sha256,
            backend=result.backend,
            performance=_frame_records(result.performance),
            positions=_frame_records(result.positions),
            fills=tuple(dict(record) for record in fills),
            metadata=result_metadata,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "scenario_sha256": self.scenario_sha256,
            "backend": self.backend,
            "performance": _sorted_records(self.performance),
            "positions": _sorted_records(self.positions),
            "fills": _sorted_records(self.fills),
            "metadata": _json_ready(self.metadata),
        }


def lean_scenario_sha256(scenario: LeanGoldenScenario) -> str:
    return _content_sha256(SCENARIO_SCHEMA, scenario.to_payload())


def lean_result_sha256(result: LeanGoldenResult) -> str:
    return _content_sha256(RESULT_SCHEMA, result.to_payload())


def export_lean_scenario(scenario: LeanGoldenScenario, path: str | Path) -> str:
    digest = lean_scenario_sha256(scenario)
    _write_envelope(path, SCENARIO_SCHEMA, scenario.to_payload(), digest)
    return digest


def export_lean_result(result: LeanGoldenResult, path: str | Path) -> str:
    digest = lean_result_sha256(result)
    _write_envelope(path, RESULT_SCHEMA, result.to_payload(), digest)
    return digest


def load_lean_scenario(path: str | Path) -> LeanGoldenScenario:
    payload = _load_envelope(path, SCENARIO_SCHEMA)
    return LeanGoldenScenario(
        scenario_id=_required_text(payload, "scenario_id"),
        description=_required_text(payload, "description"),
        positions=_record_tuple(payload, "positions"),
        pricing=_record_tuple(payload, "pricing"),
        periods=_record_tuple(payload, "periods"),
        config=_required_mapping(payload, "config"),
        metadata=_required_mapping(payload, "metadata"),
    )


def load_lean_result(path: str | Path) -> LeanGoldenResult:
    payload = _load_envelope(path, RESULT_SCHEMA)
    return LeanGoldenResult(
        scenario_sha256=_required_text(payload, "scenario_sha256"),
        backend=_required_text(payload, "backend"),
        performance=_record_tuple(payload, "performance"),
        positions=_record_tuple(payload, "positions"),
        fills=_record_tuple(payload, "fills"),
        metadata=_required_mapping(payload, "metadata"),
    )


def _frame_records(frame: pd.DataFrame) -> tuple[Mapping[str, Any], ...]:
    return tuple(dict(record) for record in frame.to_dict("records"))


def _sorted_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized = [_json_ready(dict(record)) for record in records]
    return sorted(normalized, key=_canonical_json)


def _json_ready(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if not math.isfinite(value):
            raise ValueError("LEAN golden exchange does not support infinite values.")
        return value
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_ready(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        return _json_ready(item())
    raise TypeError(f"Value of type {type(value).__name__} is not JSON serializable.")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _json_ready(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _content_sha256(schema: str, payload: Mapping[str, Any]) -> str:
    content = _canonical_json({"schema": schema, "payload": payload}).encode()
    return hashlib.sha256(content).hexdigest()


def _write_envelope(
    path: str | Path,
    schema: str,
    payload: Mapping[str, Any],
    digest: str,
) -> None:
    envelope = {"schema": schema, "content_sha256": digest, "payload": payload}
    Path(path).write_text(_canonical_json(envelope) + "\n", encoding="utf-8")


def _load_envelope(path: str | Path, expected_schema: str) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise LeanGoldenExchangeError("LEAN golden envelope must be a JSON object.")
    if raw.get("schema") != expected_schema:
        raise LeanGoldenExchangeError(
            f"Expected schema {expected_schema!r}, got {raw.get('schema')!r}."
        )
    payload = raw.get("payload")
    if not isinstance(payload, dict):
        raise LeanGoldenExchangeError("LEAN golden envelope payload must be an object.")
    expected_digest = _content_sha256(expected_schema, payload)
    if raw.get("content_sha256") != expected_digest:
        raise LeanGoldenExchangeError("LEAN golden envelope content hash does not match.")
    return payload


def _required_text(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise LeanGoldenExchangeError(f"LEAN golden payload field {key!r} must be text.")
    return value


def _required_mapping(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise LeanGoldenExchangeError(f"LEAN golden payload field {key!r} must be an object.")
    return value


def _record_tuple(payload: Mapping[str, Any], key: str) -> tuple[Mapping[str, Any], ...]:
    value = payload.get(key)
    if not isinstance(value, list) or not all(isinstance(record, dict) for record in value):
        raise LeanGoldenExchangeError(f"LEAN golden payload field {key!r} must be record objects.")
    return tuple(value)


__all__ = [
    "LeanGoldenExchangeError",
    "LeanGoldenResult",
    "LeanGoldenScenario",
    "export_lean_result",
    "export_lean_scenario",
    "lean_result_sha256",
    "lean_scenario_sha256",
    "load_lean_result",
    "load_lean_scenario",
]
