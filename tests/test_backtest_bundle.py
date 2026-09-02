from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from portfolio_backtester.backtest_bundle import (
    BacktestBundleInventoryItem,
    BacktestBundleManifest,
    BacktestEvidenceTier,
    reconcile_unified_ledger,
    validate_execution_aware_bundle_inputs,
)
from portfolio_backtester.backtest_bundle_io import (
    read_backtest_bundle,
    write_backtest_bundle,
)
from portfolio_backtester.execution_sim.results import UnifiedLedger

SHA_A = "a" * 64


def ledger() -> UnifiedLedger:
    return UnifiedLedger(
        targets=pd.DataFrame(
            {
                "rebalance_date": ["2026-09-02"],
                "symbol": ["000001.SZ"],
                "target_weight": [1.0],
            }
        ),
        orders=pd.DataFrame({"order_id": ["o1"], "status": ["filled"]}),
        fills=pd.DataFrame({"fill_id": ["f1"], "order_id": ["o1"]}),
        daily_positions=pd.DataFrame(
            {"trade_date": ["2026-09-03"], "positions_value": [800.0]}
        ),
        daily_cash=pd.DataFrame({"trade_date": ["2026-09-03"], "cash": [200.0]}),
        daily_nav=pd.DataFrame({"trade_date": ["2026-09-03"], "nav": [1000.0]}),
        cost_breakdown=pd.DataFrame({"side": ["total"], "transaction_cost": [1.0]}),
        turnover_breakdown=pd.DataFrame(
            {"side": ["total"], "filled_notional": [800.0]}
        ),
    )


def execution_clock() -> dict[str, str]:
    return {
        "schema_version": "research.clock.v1",
        "timezone": "Asia/Shanghai",
        "information_cutoff_at": "2026-09-02T15:00:00+08:00",
        "signal_at": "2026-09-02T15:01:00+08:00",
        "decision_at": "2026-09-02T15:02:00+08:00",
        "earliest_order_at": "2026-09-03T09:15:00+08:00",
        "execution_window_start_at": "2026-09-03T09:30:00+08:00",
        "execution_window_end_at": "2026-09-03T10:00:00+08:00",
        "valuation_at": "2026-09-03T15:00:00+08:00",
        "timing_policy_id": "a-share.close-next-open.v1",
        "trading_calendar_ref": "sse-szse-20260902",
    }


def capabilities() -> dict[str, object]:
    return {
        "order_lifecycle": True,
        "daily_ledger": True,
        "partial_fills": True,
    }


def inventory_payload() -> list[dict[str, object]]:
    names = [
        "targets.parquet",
        "orders.parquet",
        "fills.parquet",
        "daily_positions.parquet",
        "daily_cash.parquet",
        "daily_nav.parquet",
        "cost_breakdown.parquet",
        "turnover_breakdown.parquet",
        "diagnostics.json",
    ]
    return [
        {
            "path": name,
            "sha256": SHA_A,
            "required": True,
            "rows": 1 if name.endswith(".parquet") else None,
        }
        for name in names
    ]


def manifest_payload() -> dict[str, object]:
    return {
        "schema_version": "portfolio_backtester.backtest_result.v1",
        "run_id": "run-001",
        "evidence_tier": "execution_aware",
        "artifact_envelope": {
            "schema_version": "research.artifact-envelope.v2",
            "content_sha256": SHA_A,
        },
        "research_clock": execution_clock(),
        "backend": {"name": "native.position_replay", "version": "0.4.0"},
        "backend_capabilities": capabilities(),
        "input_refs": [{"artifact_id": "signals", "sha256": SHA_A}],
        "inventory": inventory_payload(),
        "reconciliation": {"status": "passed", "rows": 1, "max_abs_error": 0.0},
    }


def writer_kwargs() -> dict[str, object]:
    return {
        "run_id": "run-001",
        "evidence_tier": BacktestEvidenceTier.EXECUTION_AWARE,
        "ledger": ledger(),
        "research_clock": execution_clock(),
        "backend": {"name": "native.position_replay", "version": "0.4.0"},
        "backend_capabilities": capabilities(),
        "producer": {
            "repository": "portfolio-backtester",
            "version": "0.4.0",
            "commit": "abc123",
            "backend": "native.position_replay",
        },
        "configuration_sha256": SHA_A,
        "input_refs": [{"artifact_id": "signals", "sha256": SHA_A}],
        "diagnostics": {"note": "fixture"},
        "created_at": datetime(2026, 9, 3, 16, 0, tzinfo=timezone.utc),
    }


def test_evidence_tier_rejects_unknown_value():
    with pytest.raises(ValueError):
        BacktestEvidenceTier("trust_me")


def test_reconcile_unified_ledger_passes_balanced_account():
    report = reconcile_unified_ledger(ledger())
    assert report["status"] == "passed"
    assert report["rows"] == 1
    assert report["max_abs_error"] == 0.0


def test_reconcile_unified_ledger_rejects_unbalanced_nav():
    broken = ledger()
    broken.daily_nav.loc[0, "nav"] = 999.0
    with pytest.raises(ValueError, match=r"nav = cash \+ positions_value"):
        reconcile_unified_ledger(broken)


def test_execution_aware_requires_order_and_daily_ledger_capabilities():
    caps = capabilities()
    caps["order_lifecycle"] = False
    with pytest.raises(ValueError, match="order_lifecycle"):
        validate_execution_aware_bundle_inputs(
            ledger(), research_clock=execution_clock(), backend_capabilities=caps
        )


def test_execution_aware_requires_complete_execution_clock():
    clock = execution_clock()
    del clock["execution_window_start_at"]
    with pytest.raises(ValueError, match="execution_window_start_at"):
        validate_execution_aware_bundle_inputs(
            ledger(), research_clock=clock, backend_capabilities=capabilities()
        )


def test_backtest_bundle_manifest_round_trip():
    manifest = BacktestBundleManifest.from_mapping(manifest_payload())
    restored = BacktestBundleManifest.from_mapping(manifest.to_mapping())
    assert restored == manifest
    assert isinstance(manifest.inventory[0], BacktestBundleInventoryItem)


def test_execution_aware_manifest_requires_all_ledger_files():
    payload = manifest_payload()
    payload["inventory"] = [
        item for item in payload["inventory"] if item["path"] != "fills.parquet"
    ]
    with pytest.raises(
        ValueError,
        match="missing required execution-aware bundle files: fills.parquet",
    ):
        BacktestBundleManifest.from_mapping(payload)


def test_inventory_item_rejects_unsafe_path():
    with pytest.raises(ValueError, match="safe relative path"):
        BacktestBundleInventoryItem.from_mapping(
            {
                "path": "../outside.parquet",
                "sha256": SHA_A,
                "required": True,
                "rows": 1,
            }
        )


def test_execution_aware_manifest_requires_passed_reconciliation():
    payload = manifest_payload()
    payload["reconciliation"] = {"status": "failed"}
    with pytest.raises(ValueError, match="requires passed reconciliation"):
        BacktestBundleManifest.from_mapping(payload)


def test_writer_cleans_up_when_frame_serialization_fails(tmp_path: Path, monkeypatch):
    def fail_to_parquet(self, path, *, index=False):
        raise RuntimeError("parquet boom")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_to_parquet)
    output = tmp_path / "backtest_result"
    with pytest.raises(RuntimeError, match="parquet boom"):
        write_backtest_bundle(output, **writer_kwargs())
    assert not output.exists()
    assert not list(tmp_path.glob(".backtest_result.tmp-*"))


def test_writer_refuses_to_overwrite_existing_bundle(tmp_path: Path):
    output = tmp_path / "backtest_result"
    output.mkdir()
    with pytest.raises(FileExistsError):
        write_backtest_bundle(output, **writer_kwargs())


def test_write_read_bundle_round_trip_and_hash_verification(tmp_path: Path):
    pytest.importorskip("pyarrow")
    output = tmp_path / "backtest_result"
    written = write_backtest_bundle(output, **writer_kwargs())
    restored = read_backtest_bundle(output)
    assert restored == written
    assert (output / "manifest.json").is_file()
    assert (output / "daily_nav.parquet").is_file()

    (output / "diagnostics.json").write_text('{"tampered":true}', encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        read_backtest_bundle(output)


def test_execution_aware_manifest_rejects_incomplete_clock_on_read():
    payload = manifest_payload()
    del payload["research_clock"]["earliest_order_at"]
    with pytest.raises(ValueError, match="research_clock.earliest_order_at"):
        BacktestBundleManifest.from_mapping(payload)


def test_execution_aware_manifest_rejects_false_backend_capability_on_read():
    payload = manifest_payload()
    payload["backend_capabilities"]["daily_ledger"] = False
    with pytest.raises(ValueError, match="backend capability daily_ledger"):
        BacktestBundleManifest.from_mapping(payload)
