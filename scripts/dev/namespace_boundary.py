#!/usr/bin/env python3
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
PACKAGE = SRC / "portfolio_backtester"
FORBIDDEN_OWNERS = ("strategy_pipeline", "alpha_research")
LEGACY_BRAND_MARKERS = (
    "".join(("cs", "tree")),
    "".join(("cross", "_sectional", "_trees")),
    "".join(("cross", "-sectional", "-trees")),
)
SHARED_NAMESPACE_MARKER = "pkgutil import " + "extend_path"
TEXT_SUFFIXES = {".json", ".md", ".py", ".sh", ".toml", ".yaml", ".yml"}
TEXT_SURFACES = (
    ROOT / "README.md",
    ROOT / "pyproject.toml",
    ROOT / "docs",
    ROOT / "scripts",
    ROOT / "src" / "portfolio_backtester",
    ROOT / "tests",
)


def _escapes_owner(path: Path, level: int) -> bool:
    package_depth = len(path.relative_to(SRC).parent.parts)
    return level > package_depth


def _surface_files() -> list[Path]:
    files: list[Path] = []
    for surface in TEXT_SURFACES:
        if surface.is_file():
            files.append(surface)
            continue
        files.extend(
            path for path in surface.rglob("*") if path.is_file() and path.suffix in TEXT_SUFFIXES
        )
    return sorted(files)


def main() -> int:
    offenders: list[str] = []
    for path in _surface_files():
        relative = path.relative_to(ROOT)
        relative_text = relative.as_posix().casefold()
        text = path.read_text(encoding="utf-8")
        folded_text = text.casefold()
        for marker in LEGACY_BRAND_MARKERS:
            if marker in relative_text or marker in folded_text:
                offenders.append(f"legacy brand marker: {relative}")
                break
        if SHARED_NAMESPACE_MARKER in text:
            offenders.append(f"shared namespace mechanism: {relative}")

    for path in sorted(PACKAGE.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        relative = path.relative_to(ROOT)
        tree = ast.parse(text, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(FORBIDDEN_OWNERS):
                        offenders.append(
                            f"cross-owner import: {relative}:{node.lineno}:{alias.name}"
                        )
            elif isinstance(node, ast.ImportFrom):
                if node.level and _escapes_owner(path, node.level):
                    offenders.append(
                        "relative import escapes owner: "
                        f"{relative}:{node.lineno}:level={node.level}"
                    )
                module = node.module or ""
                if module.startswith(FORBIDDEN_OWNERS):
                    offenders.append(f"cross-owner import: {relative}:{node.lineno}:{module}")

    if offenders:
        raise SystemExit("\n".join(offenders))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
