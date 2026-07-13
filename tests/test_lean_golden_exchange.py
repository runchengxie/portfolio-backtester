from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from cstree.backtesting.backends import NativeAShareReplayBackend, PositionReplayRequest
from cstree.backtesting.integrations.lean import (
    LeanGoldenExchangeError,
    LeanGoldenResult,
    LeanGoldenScenario,
    export_lean_result,
    export_lean_scenario,
    lean_scenario_sha256,
    load_lean_result,
    load_lean_scenario,
)

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "lean"


def _scenario(request: PositionReplayRequest) -> LeanGoldenScenario:
    return LeanGoldenScenario.from_position_replay_request(
        scenario_id="a-share-delayed-exit",
        description="A suspended symbol exits one day late and incurs A-share replay fees.",
        request=request,
        metadata={"market": "CN", "reference_runtime": "LEAN"},
    )


def test_scenario_hash_is_stable_across_record_order(
    a_share_replay_request: PositionReplayRequest,
) -> None:
    scenario = _scenario(a_share_replay_request)
    reordered = replace(scenario, positions=tuple(reversed(scenario.positions)))

    assert lean_scenario_sha256(reordered) == lean_scenario_sha256(scenario)


def test_scenario_and_result_round_trip_without_lean_runtime(
    tmp_path: Path,
    a_share_replay_request: PositionReplayRequest,
) -> None:
    scenario = _scenario(a_share_replay_request)
    scenario_path = tmp_path / "scenario.json"
    scenario_hash = export_lean_scenario(scenario, scenario_path)
    native = NativeAShareReplayBackend().run(a_share_replay_request)
    fills = (
        {
            "date": "2020-01-02",
            "symbol": "AAA",
            "side": "buy",
            "weight": 0.5,
            "price": 10.0,
        },
        {
            "date": "2020-01-04",
            "symbol": "AAA",
            "side": "sell",
            "weight": 0.5,
            "price": 12.0,
            "reason": "delayed_untradable_exit",
        },
    )
    golden_result = LeanGoldenResult.from_backend_result(
        scenario_sha256=scenario_hash,
        result=native,
        fills=fills,
        metadata={"runtime_version": "external"},
    )
    result_path = tmp_path / "result.json"

    export_lean_result(golden_result, result_path)

    assert load_lean_scenario(scenario_path).to_payload() == scenario.to_payload()
    loaded_result = load_lean_result(result_path)
    assert loaded_result.to_payload() == golden_result.to_payload()
    assert loaded_result.fills[1]["reason"] == "delayed_untradable_exit"


def test_exchange_rejects_modified_payload(
    tmp_path: Path,
    a_share_replay_request: PositionReplayRequest,
) -> None:
    path = tmp_path / "scenario.json"
    export_lean_scenario(_scenario(a_share_replay_request), path)
    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["payload"]["description"] = "modified"
    path.write_text(json.dumps(envelope), encoding="utf-8")

    with pytest.raises(LeanGoldenExchangeError, match="content hash"):
        load_lean_scenario(path)


def test_committed_a_share_golden_files_match_native_semantics(
    a_share_replay_request: PositionReplayRequest,
) -> None:
    scenario = _scenario(a_share_replay_request)
    fixture_scenario = load_lean_scenario(FIXTURE_ROOT / "a_share_delayed_exit.scenario.json")
    fixture_result = load_lean_result(FIXTURE_ROOT / "a_share_delayed_exit.native-result.json")
    native = NativeAShareReplayBackend().run(a_share_replay_request)

    assert fixture_scenario.to_payload() == scenario.to_payload()
    assert fixture_result.scenario_sha256 == lean_scenario_sha256(scenario)
    current_payload = LeanGoldenResult.from_backend_result(
        scenario_sha256=fixture_result.scenario_sha256,
        result=native,
    ).to_payload()
    assert list(fixture_result.performance) == current_payload["performance"]
    assert fixture_result.metadata["source_backend_metadata"] == native.metadata
    assert fixture_result.fills[-1]["reason"] == "delayed_untradable_exit"
