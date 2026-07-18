from __future__ import annotations

from pathlib import Path

import yaml

LEDGER_PATH = Path(__file__).parents[1] / "docs" / "framework-integration-ledger.yml"


def test_framework_integration_ledger_has_explicit_backend_boundaries() -> None:
    ledger = yaml.safe_load(LEDGER_PATH.read_text(encoding="utf-8"))

    assert ledger["schema_version"] == "portfolio_backtester.framework_integration.v1"
    assert ledger["policy"]["native_default_until_parity"] is True
    assert ledger["backends"]["native"]["status"] == "canonical"
    assert ledger["backends"]["backtrader"]["status"] == "evaluation"
    assert ledger["backends"]["qlib"]["status"] == "evaluation"
    assert ledger["backends"]["vnpy"]["status"] == "transport_only"
    assert ledger["backends"]["vnpy"]["owner"] == "quant-execution-engine"


def test_external_backends_have_adoption_and_rollback_gates() -> None:
    ledger = yaml.safe_load(LEDGER_PATH.read_text(encoding="utf-8"))

    for name in ("backtrader", "qlib", "vnpy"):
        backend = ledger["backends"][name]
        assert backend["adoption_gates"]
        assert backend["rollback"]

    replacement = ledger["replacement_gates"]
    assert replacement["coverage_ratio_minimum"] == 0.90
    assert "all_golden_scenarios_pass" in replacement["requirements"]
