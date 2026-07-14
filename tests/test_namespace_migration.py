from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def test_owner_native_layout() -> None:
    assert (SRC / "portfolio_backtester" / "__init__.py").is_file()
    assert not (SRC / "cstree").exists()


def test_legacy_namespace_cannot_regrow() -> None:
    offenders = []
    for path in (SRC / "portfolio_backtester").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "cstree.backtesting" in text or "pkgutil import extend_path" in text:
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []
