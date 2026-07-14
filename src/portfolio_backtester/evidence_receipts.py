"""Stable receipts for portfolio sizing and risk-allocation evidence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd


def series_sha256(values: pd.Series) -> str:
    """Hash an indexed numeric series using a stable CSV representation."""

    normalized = pd.to_numeric(values, errors="coerce").fillna(0.0)
    payload = normalized.sort_index().to_csv(header=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 for an artifact file."""

    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_portfolio_sizing_receipt(
    weights: pd.Series,
    *,
    method: str,
    configuration: Mapping[str, Any] | None = None,
    source_positions: str | Path | None = None,
    calibration_artifact: str | Path | None = None,
    covariance_artifact: str | Path | None = None,
) -> dict[str, object]:
    """Describe the final portfolio weights regardless of weighting family.

    The receipt supports equal, signal, liquidity, calibrated probability, and
    risk-budget methods. It records what was produced; it does not recompute the
    portfolio or authorize execution.
    """

    cleaned = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    active = cleaned.loc[cleaned.abs() > 0]
    receipt: dict[str, object] = {
        "schema_version": 1,
        "method": str(method).strip().lower() or "unknown",
        "config": dict(configuration or {}),
        "target_count": int(len(active)),
        "gross_exposure": float(active.abs().sum()),
        "net_exposure": float(active.sum()),
        "maximum_weight": float(active.abs().max()) if not active.empty else 0.0,
        "minimum_active_weight": float(active.abs().min()) if not active.empty else 0.0,
        "weights_sha256": series_sha256(cleaned),
        "source_positions": str(Path(source_positions)) if source_positions else None,
        "calibration_artifact": (
            str(Path(calibration_artifact)) if calibration_artifact else None
        ),
        "covariance_artifact": str(Path(covariance_artifact)) if covariance_artifact else None,
    }
    if source_positions is not None:
        source = Path(source_positions).expanduser()
        receipt["source_positions_sha256"] = sha256_file(source) if source.is_file() else None
    return receipt


def write_receipt(payload: Mapping[str, object], path: str | Path) -> None:
    """Write a JSON receipt with deterministic human-readable formatting."""

    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "build_portfolio_sizing_receipt",
    "series_sha256",
    "sha256_file",
    "write_receipt",
]
