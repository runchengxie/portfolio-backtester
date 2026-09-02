from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_public_ci_runs_typecheck_before_tests() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    typecheck = ".venv/bin/ty check"
    tests = ".venv/bin/pytest -q"
    assert typecheck in workflow
    assert workflow.index(typecheck) < workflow.index(tests)
