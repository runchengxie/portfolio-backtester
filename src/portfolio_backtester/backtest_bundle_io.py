from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .backtest_bundle import (
    BACKTEST_BUNDLE_SCHEMA_VERSION,
    BacktestBundleInventoryItem,
    BacktestBundleManifest,
    BacktestEvidenceTier,
    reconcile_unified_ledger,
    validate_execution_aware_bundle_inputs,
)
from .execution_sim.results import UnifiedLedger

_FRAME_FILES = (
    ("targets.parquet", "targets"),
    ("orders.parquet", "orders"),
    ("fills.parquet", "fills"),
    ("daily_positions.parquet", "daily_positions"),
    ("daily_cash.parquet", "daily_cash"),
    ("daily_nav.parquet", "daily_nav"),
    ("cost_breakdown.parquet", "cost_breakdown"),
    ("turnover_breakdown.parquet", "turnover_breakdown"),
)


def _shared_contracts() -> tuple[Any, Any, Any, Any]:
    from research_contracts import (
        ArtifactEnvelopeV2,
        LineageInput,
        ProducerIdentity,
        canonical_json_sha256,
    )

    return ArtifactEnvelopeV2, LineageInput, ProducerIdentity, canonical_json_sha256


def _file_sha256(path: Path) -> str:
    from research_contracts import file_sha256

    return str(file_sha256(path))


def _json_payload(value: Any) -> Any:
    from .backends.base import to_json_compatible

    return to_json_compatible(value)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(
            _json_payload(payload),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _evidence_tier(value: BacktestEvidenceTier | str) -> BacktestEvidenceTier:
    if isinstance(value, BacktestEvidenceTier):
        return value
    return BacktestEvidenceTier(str(value))


def _lineage_inputs(input_refs: Sequence[Mapping[str, Any]]) -> tuple[Any, ...]:
    _, LineageInput, _, _ = _shared_contracts()
    return tuple(
        LineageInput(
            artifact_id=str(item.get("artifact_id", "")).strip(),
            sha256=str(item.get("sha256", "")).strip(),
        )
        for item in input_refs
    )


def _inventory_content_sha256(inventory: Sequence[BacktestBundleInventoryItem]) -> str:
    _, _, _, canonical_json_sha256 = _shared_contracts()
    return str(canonical_json_sha256([item.to_mapping() for item in inventory]))


def _diagnostic_reconciliation(ledger: UnifiedLedger) -> dict[str, Any]:
    try:
        return dict(reconcile_unified_ledger(ledger))
    except (TypeError, ValueError) as exc:
        return {"status": "not_available", "reason": str(exc)}


def write_backtest_bundle(
    output_dir: Path,
    *,
    run_id: str,
    evidence_tier: BacktestEvidenceTier | str,
    ledger: UnifiedLedger,
    research_clock: Mapping[str, Any],
    backend: Mapping[str, Any],
    backend_capabilities: Mapping[str, Any],
    producer: Mapping[str, Any],
    configuration_sha256: str,
    input_refs: Sequence[Mapping[str, Any]] = (),
    diagnostics: Mapping[str, Any] | None = None,
    created_at: datetime | None = None,
    artifact_id: str | None = None,
) -> BacktestBundleManifest:
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"backtest bundle already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.tmp-",
            dir=output_dir.parent,
        )
    )
    tier = _evidence_tier(evidence_tier)
    try:
        reconciliation = (
            dict(
                validate_execution_aware_bundle_inputs(
                    ledger,
                    research_clock=research_clock,
                    backend_capabilities=backend_capabilities,
                )
            )
            if tier is BacktestEvidenceTier.EXECUTION_AWARE
            else _diagnostic_reconciliation(ledger)
        )

        inventory: list[BacktestBundleInventoryItem] = []
        frames_required = tier is BacktestEvidenceTier.EXECUTION_AWARE
        for filename, field in _FRAME_FILES:
            frame = getattr(ledger, field)
            path = tmp_dir / filename
            frame.to_parquet(path, index=False)
            inventory.append(
                BacktestBundleInventoryItem(
                    path=filename,
                    sha256=_file_sha256(path),
                    required=frames_required,
                    rows=int(frame.shape[0]),
                )
            )

        diagnostics_path = tmp_dir / "diagnostics.json"
        _write_json(diagnostics_path, dict(diagnostics or {}))
        inventory.append(
            BacktestBundleInventoryItem(
                path="diagnostics.json",
                sha256=_file_sha256(diagnostics_path),
                required=True,
                rows=None,
            )
        )
        inventory_tuple = tuple(inventory)

        ArtifactEnvelopeV2, _, ProducerIdentity, _ = _shared_contracts()
        timestamp = created_at or datetime.now(timezone.utc)
        envelope = ArtifactEnvelopeV2(
            artifact_id=artifact_id or f"backtest:{run_id}",
            artifact_type=BACKTEST_BUNDLE_SCHEMA_VERSION,
            run_id=run_id,
            created_at=timestamp,
            producer=ProducerIdentity.from_mapping(producer),
            configuration_sha256=configuration_sha256,
            content_sha256=_inventory_content_sha256(inventory_tuple),
            lineage=_lineage_inputs(input_refs),
        )
        manifest = BacktestBundleManifest(
            run_id=run_id,
            evidence_tier=tier,
            artifact_envelope=envelope.to_mapping(),
            research_clock=dict(research_clock),
            backend=dict(backend),
            backend_capabilities=dict(backend_capabilities),
            input_refs=tuple(dict(item) for item in input_refs),
            inventory=inventory_tuple,
            reconciliation=reconciliation,
        )
        _write_json(tmp_dir / "manifest.json", manifest.to_mapping())
        os.replace(tmp_dir, output_dir)
        return manifest
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def read_backtest_bundle(
    path: Path,
    *,
    verify_hashes: bool = True,
) -> BacktestBundleManifest:
    root = Path(path)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"backtest bundle manifest is missing: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("backtest bundle manifest must be an object")
    manifest = BacktestBundleManifest.from_mapping(payload)

    ArtifactEnvelopeV2, _, _, _ = _shared_contracts()
    envelope = ArtifactEnvelopeV2.from_mapping(manifest.artifact_envelope)
    expected_content = _inventory_content_sha256(manifest.inventory)
    if envelope.content_sha256 != expected_content:
        raise ValueError("backtest bundle inventory SHA-256 mismatch")

    if verify_hashes:
        resolved_root = root.resolve()
        for item in manifest.inventory:
            item_path = (root / item.path).resolve()
            if resolved_root not in item_path.parents:
                raise ValueError(f"backtest bundle file escapes root: {item.path}")
            if not item_path.is_file():
                raise ValueError(f"backtest bundle file is missing: {item.path}")
            if _file_sha256(item_path) != item.sha256:
                raise ValueError(f"backtest bundle file SHA-256 mismatch: {item.path}")
    return manifest


__all__ = ["read_backtest_bundle", "write_backtest_bundle"]
