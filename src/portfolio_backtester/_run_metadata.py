"""Reproducibility metadata snapshot for portfolio-backtester runs.

This module collects a stable, auditable fingerprint of a single run so that
any report (capacity report, backtest summary, etc.) can be traced back to the
exact code, configuration, inputs, and environment that produced it.

Design notes
------------
- Every field is collected from real artifacts (git commit, file hashes,
  importlib versions, wall-clock time). Where the roadmap asks for a "version"
  that the repository does not actually track (universe version, trading-calendar
  version, fee/slippage calibration version), we derive a *fingerprint* from the
  real inputs rather than inventing a version number. This keeps the snapshot
  honest and reproducible without fabricating provenance.
- ``collect_reproducibility`` is a pure function over its arguments; the only
  side-effecting read is the optional ``git rev-parse HEAD`` (which degrades
  gracefully to ``"unknown"`` when unavailable). Nothing is written by it.
- ``write_run_metadata`` persists the snapshot next to the run artifacts using
  the same deterministic JSON writer as ``evidence_receipts``.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from portfolio_backtester.evidence_receipts import sha256_file, write_receipt

_GIT_REF_ENV = "PORTFOLIO_BACKTESTER_COMMIT"


def _repo_commit(repo_root: Path | None = None) -> str:
    """Best-effort commit hash of the portfolio-backtester source tree.

    Resolution order: explicit env override, then ``git rev-parse HEAD`` from
    the repo root (or CWD). Returns ``"unknown"`` if neither is available.
    """

    override = _GIT_REF_ENV and __import__("os").environ.get(_GIT_REF_ENV)
    if override:
        return override.strip()
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root) if repo_root else None,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def _dependency_versions() -> dict[str, str]:
    """Collect versions of portfolio-backtester and its key dependencies."""

    from importlib.metadata import PackageNotFoundError, version

    packages = ["portfolio-backtester", "pandas", "numpy"]
    collected: dict[str, str] = {}
    for name in packages:
        try:
            collected[name] = version(name)
        except PackageNotFoundError:
            collected[name] = "unknown"
    return collected


def _config_version_field(config: Mapping[str, Any] | None, key: str) -> str:
    """Extract a named version-ish field from the run config if present.

    Returns ``"unversioned"`` when the config or the key is missing, so the
    snapshot stays truthful instead of inventing a value.
    """

    if not config:
        return "unversioned"
    value = config.get(key)
    if value in (None, ""):
        return "unversioned"
    return str(value)


def collect_reproducibility(
    *,
    config_path: Path,
    positions_path: Path,
    pricing_path: Path,
    run_dir: Path,
    market: str,
    backend_name: str | None = None,
    capability_snapshot: Mapping[str, Any] | None = None,
    random_seed: int | None = None,
    calendar_window: Mapping[str, str | None] | None = None,
    config: Mapping[str, Any] | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Build a reproducibility snapshot dict for a single run.

    The returned mapping is JSON-serializable and stable in key order. It is
    safe to add as a top-level ``reproducibility`` field on any report payload.
    """

    config_path = Path(config_path)
    positions_path = Path(positions_path)
    pricing_path = Path(pricing_path)
    run_dir = Path(run_dir)

    config_hash = sha256_file(config_path) if config_path.is_file() else "missing"
    positions_hash = sha256_file(positions_path) if positions_path.is_file() else "missing"
    pricing_hash = sha256_file(pricing_path) if pricing_path.is_file() else "missing"

    # Input-data fingerprint: combine both input hashes so any change to either
    # input file changes the combined digest.
    combined = hashlib.sha256()
    combined.update(positions_hash.encode("utf-8"))
    combined.update(pricing_hash.encode("utf-8"))
    input_data_hash = combined.hexdigest()

    # Universe fingerprint reuses the positions hash: the stock pool is exactly
    # the set of symbols in the positions file. This is a derived fingerprint,
    # not a fabricated semantic version.
    universe_fingerprint = positions_hash

    return {
        "schema": "reproducibility.v1",
        "repo_commit": _repo_commit(repo_root),
        "market": market,
        "backend_name": backend_name,
        "capability_snapshot": dict(capability_snapshot) if capability_snapshot else None,
        "config_hash": config_hash,
        "input_data_hash": input_data_hash,
        "positions_hash": positions_hash,
        "pricing_hash": pricing_hash,
        "universe_fingerprint": universe_fingerprint,
        "calendar_window": dict(calendar_window) if calendar_window else None,
        "fee_schedule_version": _config_version_field(config, "fee_schedule_version"),
        "slippage_calibration_version": _config_version_field(
            config, "slippage_calibration_version"
        ),
        "dependency_versions": _dependency_versions(),
        "random_seed": random_seed,
        "run_timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
    }


def write_run_metadata(reproducibility: Mapping[str, Any], run_dir: Path) -> Path:
    """Persist the reproducibility snapshot as ``run_metadata.json`` in run_dir."""

    target = Path(run_dir) / "run_metadata.json"
    write_receipt(reproducibility, target)
    return target


def _normalize_payload(value: Any) -> Any:
    """Helper kept for callers that want a JSON-safe copy (defensive)."""

    return json.loads(json.dumps(value, default=str))


__all__ = [
    "collect_reproducibility",
    "write_run_metadata",
]
