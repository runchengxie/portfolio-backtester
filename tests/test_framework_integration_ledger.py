from __future__ import annotations

from pathlib import Path

import yaml

from portfolio_backtester.backends import NativePositionReplayBackend

LEDGER_PATH = Path(__file__).parents[1] / "docs" / "framework-integration-ledger.yml"


def test_framework_integration_ledger_matches_main_branch_backends() -> None:
    ledger = yaml.safe_load(LEDGER_PATH.read_text(encoding="utf-8"))

    assert ledger["schema_version"] == "portfolio_backtester.framework_integration.v2"
    assert ledger["policy"]["native_default_until_parity"] is True
    assert ledger["policy"]["current_registry_only"] == [NativePositionReplayBackend.name]
    assert set(ledger["backends"]) == {"native"}
    assert ledger["backends"]["native"]["status"] == "canonical"
    assert ledger["backends"]["native"]["registry_name"] == NativePositionReplayBackend.name


def test_framework_history_and_plans_are_not_registered_backends() -> None:
    ledger = yaml.safe_load(LEDGER_PATH.read_text(encoding="utf-8"))

    assert ledger["historical_candidates"]["qlib"]["status"] == "not_merged_to_main"
    assert ledger["historical_candidates"]["qlib"]["current_role"] == "none"
    assert ledger["historical_candidates"]["lean"]["status"] == "not_merged_to_main"
    assert ledger["historical_candidates"]["lean"]["current_role"] == "reference_only"

    backtrader = ledger["planned_integrations"]["backtrader"]
    assert backtrader["status"] == "planning_only"
    assert backtrader["adoption_gates"]
    assert backtrader["rollback"]

    assert ledger["out_of_scope"]["vnpy"]["status"] == "out_of_scope"

    replacement = ledger["replacement_gates"]
    assert replacement["coverage_ratio_minimum"] == 0.90
    assert "all_golden_scenarios_pass" in replacement["requirements"]


def test_main_branch_has_no_external_framework_adapter_or_dependency() -> None:
    root = LEDGER_PATH.parents[1]
    backends_dir = root / "src" / "portfolio_backtester" / "backends"
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8").lower()

    for framework in ("qlib", "lean", "backtrader", "vnpy"):
        assert not (backends_dir / f"{framework}.py").exists()
        assert framework not in pyproject
