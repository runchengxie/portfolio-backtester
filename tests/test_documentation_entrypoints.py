from __future__ import annotations

import re
from pathlib import Path

import portfolio_backtester
from portfolio_backtester import execution_sim

ROOT = Path(__file__).resolve().parents[1]
FACT_DOCS = (
    ROOT / "docs" / "concepts" / "cost-breakdown.md",
    ROOT / "docs" / "guides" / "execution-simulation.md",
    ROOT / "docs" / "reference" / "public-api.md",
)
STYLE_DOCS = (
    ROOT / "README.md",
    ROOT / "AGENTS.md",
    *sorted((ROOT / "docs").rglob("*.md")),
)
STYLE_PATTERNS = (
    re.compile(r"不是.{0,40}而是"),
    re.compile(r"并非.{0,40}而是"),
    re.compile(r"\*\*"),
    re.compile("\uff1b"),
    re.compile("\u2014\u2014"),
    re.compile("[\u201c\u201d]"),
)


def test_docs_use_concise_chinese_style() -> None:
    offenders: list[str] = []

    for path in STYLE_DOCS:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            for pattern in STYLE_PATTERNS:
                if pattern.search(line):
                    offenders.append(f"{path.relative_to(ROOT)}:{line_number}:{pattern.pattern}")

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

    assert "当前仓库没有启用 GitHub Actions 远端测试" in docs
    assert "本地质量门禁" in docs
    assert ".github/workflows/tests.yml" not in docs


def test_public_api_docs_cover_root_exports_and_execution_sim_surface() -> None:
    public_api_docs = FACT_DOCS[2].read_text(encoding="utf-8")
    execution_docs = FACT_DOCS[1].read_text(encoding="utf-8")

    for name in portfolio_backtester.__all__:
        assert f"`{name}`" in public_api_docs
    for name in execution_sim.__all__:
        assert f"`{name}`" in execution_docs


def test_docs_record_current_cost_and_position_limitations() -> None:
    cost_docs = FACT_DOCS[0].read_text(encoding="utf-8")
    execution_cost_docs = (ROOT / "docs" / "concepts" / "execution-costs.md").read_text(
        encoding="utf-8"
    )
    positions_docs = (ROOT / "docs" / "reference" / "outputs" / "positions.md").read_text(
        encoding="utf-8"
    )

    assert "内置滑点会进入 `fee_cost`" in cost_docs
    assert "buy_slippage_bps` 与 `sell_slippage_bps` 设为 0" in cost_docs
    assert "买卖各 10 个基点" in execution_cost_docs
    assert "`long_only=False` 不会启用空头回放" in positions_docs


def test_docs_record_daily_watch20_compatibility_exception_and_index_new_pages() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    index = (ROOT / "docs" / "README.md").read_text(encoding="utf-8")

    assert "`DailyWatch20` 是现有调用方使用的兼容例外" in readme
    assert "`DailyWatch20` 是为现有调用方保留的兼容例外" in agents
    assert "guides/execution-simulation.md" in index
    assert "concepts/afml-sizing-and-risk.md" in index
