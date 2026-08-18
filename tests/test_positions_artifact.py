from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from research_contracts import (
    ArtifactEnvelopeV2,
    canonical_json_sha256,
    file_sha256,
    read_artifact_envelope,
)

from portfolio_backtester.positions_artifact import (
    CANONICAL_POSITIONS_BY_REBALANCE_FILE,
    CANONICAL_POSITIONS_BY_REBALANCE_META_FILE,
    build_positions_envelope_v2,
    write_positions_by_rebalance_artifact,
)


def _positions_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rebalance_date": ["20260105", "20260105"],
            "entry_date": ["20260106", "20260106"],
            "symbol": ["600519.SH", "000001.SZ"],
            "weight": [0.6, 0.4],
            "signal": [1.2, 0.8],
            "rank": [1, 2],
            "side": ["long", "long"],
        }
    )


def test_write_positions_by_rebalance_writes_csv_and_readable_envelope(tmp_path: Path) -> None:
    csv_path, meta_path = write_positions_by_rebalance_artifact(
        _positions_frame(),
        tmp_path,
        run_id="run-demo",
        configuration={"top_k": 20, "weighting": "equal"},
        lineage=[("signals.parquet", "c" * 64)],
    )

    assert csv_path == tmp_path / CANONICAL_POSITIONS_BY_REBALANCE_FILE
    assert meta_path == tmp_path / CANONICAL_POSITIONS_BY_REBALANCE_META_FILE
    assert csv_path.is_file()
    assert meta_path.is_file()

    csv_frame = pd.read_csv(csv_path, dtype={"rebalance_date": str, "symbol": str})
    assert csv_frame["rebalance_date"].tolist() == ["20260105", "20260105"]
    assert csv_frame["symbol"].tolist() == ["600519.SH", "000001.SZ"]

    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    envelope = read_artifact_envelope(payload, allow_legacy=False)

    assert isinstance(envelope, ArtifactEnvelopeV2)
    assert envelope.run_id == "run-demo"
    assert envelope.artifact_id == "positions_by_rebalance:run-demo"
    assert envelope.artifact_type == "positions_by_rebalance.csv"
    assert envelope.created_at.utcoffset() is not None
    assert envelope.producer.repository == "portfolio-backtester"
    assert envelope.producer.backend == "native"
    assert envelope.content_sha256 == file_sha256(csv_path)
    assert envelope.configuration_sha256 == canonical_json_sha256(
        {"top_k": 20, "weighting": "equal"}
    )
    assert len(envelope.lineage) == 1
    assert envelope.lineage[0].artifact_id == "signals.parquet"
    assert envelope.lineage[0].sha256 == "c" * 64
    assert payload["artifact_type"] == "portfolio_backtester.positions_by_rebalance"
    assert payload["schema_version"] == 1


def test_build_positions_envelope_round_trips() -> None:
    envelope = build_positions_envelope_v2(
        run_id="run-demo",
        content_sha256="b" * 64,
        configuration={"top_k": 20},
        lineage=[("signals_style_replica.parquet", "d" * 64)],
    )

    result = ArtifactEnvelopeV2.from_mapping(envelope.to_mapping())

    assert result == envelope
    assert result.artifact_id == "positions_by_rebalance:run-demo"
    assert result.created_at.utcoffset() is not None


def test_write_positions_by_rebalance_rejects_invalid_frame(tmp_path: Path) -> None:
    invalid = _positions_frame().drop(columns=["weight"])

    with pytest.raises(ValueError, match="Invalid positions_by_rebalance frame"):
        write_positions_by_rebalance_artifact(invalid, tmp_path, run_id="run-demo")
