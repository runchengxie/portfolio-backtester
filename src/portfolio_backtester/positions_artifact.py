"""Canonical positions_by_rebalance artifact writer with artifact envelope v2."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Any

import pandas as pd
from research_contracts import (
    ArtifactEnvelopeV2,
    LineageInput,
    ProducerIdentity,
    attach_artifact_envelope_v2,
    canonical_json_sha256,
    file_sha256,
)

from .contracts import (
    CANONICAL_POSITIONS_BY_REBALANCE_FILE,
    POSITIONS_BY_REBALANCE_CONTRACT_NAME,
    POSITIONS_BY_REBALANCE_REQUIRED_COLUMNS,
    POSITIONS_BY_REBALANCE_SCHEMA_VERSION,
    assert_positions_by_rebalance_frame,
)

CANONICAL_POSITIONS_BY_REBALANCE_META_FILE = "positions_by_rebalance.meta.json"

PRODUCER_REPOSITORY = "portfolio-backtester"
PRODUCER_BACKEND = "native"


def _producer_version() -> str:
    try:
        return package_version(PRODUCER_REPOSITORY)
    except PackageNotFoundError:
        return "0.0.0"


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def build_positions_envelope_v2(
    *,
    run_id: str,
    content_sha256: str,
    configuration: Mapping[str, Any],
    lineage: Sequence[tuple[str, str]] = (),
) -> ArtifactEnvelopeV2:
    """Build a research.artifact-envelope.v2 for a positions_by_rebalance write."""
    return ArtifactEnvelopeV2(
        artifact_id=f"positions_by_rebalance:{run_id}",
        artifact_type=CANONICAL_POSITIONS_BY_REBALANCE_FILE,
        run_id=run_id,
        created_at=datetime.now(UTC),
        producer=ProducerIdentity(
            repository=PRODUCER_REPOSITORY,
            version=_producer_version(),
            commit=_git_commit(),
            backend=PRODUCER_BACKEND,
        ),
        configuration_sha256=canonical_json_sha256(configuration),
        content_sha256=content_sha256,
        lineage=tuple(LineageInput(artifact_id=item[0], sha256=item[1]) for item in lineage),
    )


def write_positions_by_rebalance_artifact(
    positions: pd.DataFrame,
    output_dir: str | Path,
    *,
    run_id: str,
    configuration: Mapping[str, Any] | None = None,
    lineage: Sequence[tuple[str, str]] = (),
    file_name: str = CANONICAL_POSITIONS_BY_REBALANCE_FILE,
) -> tuple[Path, Path]:
    """Write ``positions_by_rebalance.csv`` and its companion meta JSON.

    The companion ``positions_by_rebalance.meta.json`` carries a
    ``research.artifact-envelope.v2`` envelope under the ``artifact_envelope``
    key. The envelope content hash is the SHA-256 of the written CSV.

    Args:
        positions: Canonical positions frame to write.
        output_dir: Directory for the CSV and companion meta JSON.
        run_id: Run identifier recorded in the artifact envelope.
        configuration: Producer configuration hashed into the envelope.
        lineage: Upstream (artifact_id, sha256) pairs recorded in the envelope.
        file_name: Positions CSV file name; defaults to the canonical name.

    Returns:
        ``(csv_path, meta_path)``.
    """
    assert_positions_by_rebalance_frame(positions)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / file_name
    positions.to_csv(csv_path, index=False)
    meta_path = out_dir / CANONICAL_POSITIONS_BY_REBALANCE_META_FILE
    envelope = build_positions_envelope_v2(
        run_id=run_id,
        content_sha256=file_sha256(csv_path),
        configuration=dict(configuration or {}),
        lineage=lineage,
    )
    meta_payload = {
        "artifact_type": POSITIONS_BY_REBALANCE_CONTRACT_NAME,
        "schema_version": POSITIONS_BY_REBALANCE_SCHEMA_VERSION,
        "file": file_name,
        "metadata_file": CANONICAL_POSITIONS_BY_REBALANCE_META_FILE,
        "rows": len(positions),
        "required_columns": list(POSITIONS_BY_REBALANCE_REQUIRED_COLUMNS),
    }
    meta_payload = attach_artifact_envelope_v2(meta_payload, envelope)
    meta_path.write_text(json.dumps(meta_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return csv_path, meta_path


__all__ = [
    "CANONICAL_POSITIONS_BY_REBALANCE_META_FILE",
    "PRODUCER_BACKEND",
    "PRODUCER_REPOSITORY",
    "build_positions_envelope_v2",
    "write_positions_by_rebalance_artifact",
]
