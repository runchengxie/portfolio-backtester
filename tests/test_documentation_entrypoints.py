from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENTRY_DOCS = (
    ROOT / "README.md",
    ROOT / "AGENTS.md",
    ROOT / "docs" / "README.md",
    ROOT / "docs" / "testing.md",
)
FORBIDDEN_FRAGMENTS = ("不是", "而是", "**", "；", "——", "“", "”")


def test_entry_docs_use_concise_chinese_style() -> None:
    offenders: list[str] = []

    for path in ENTRY_DOCS:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            for fragment in FORBIDDEN_FRAGMENTS:
                if fragment in line:
                    offenders.append(f"{path.relative_to(ROOT)}:{line_number}:{fragment}")

    assert offenders == []


def test_testing_docs_match_script_modes() -> None:
    script = (ROOT / "scripts" / "dev" / "run_tests.sh").read_text(encoding="utf-8")
    docs = (ROOT / "docs" / "testing.md").read_text(encoding="utf-8")

    for mode in (
        "all",
        "fast",
        "unit",
        "lint",
        "format",
        "typecheck",
        "basedpyright",
        "maintainability",
    ):
        assert f"`{mode}`" in docs
        assert mode in script


def test_docs_record_current_automation_status() -> None:
    docs = (ROOT / "docs" / "testing.md").read_text(encoding="utf-8")

    assert "当前仓库没有启用 GitHub Actions 测试 workflow" in docs
    assert ".github/workflows/tests.yml" not in docs
