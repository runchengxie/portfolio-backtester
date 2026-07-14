#!/usr/bin/env python3
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
PACKAGE = SRC / "portfolio_backtester"
FORBIDDEN_OWNERS = ("strategy_pipeline", "alpha_research")


def _escapes_owner(path: Path, level: int) -> bool:
    package_depth = len(path.relative_to(SRC).parent.parts)
    return level > package_depth


def main() -> int:
    offenders: list[str] = []
    for path in sorted((SRC / "cstree").rglob("*.py")):
        relative = path.relative_to(ROOT)
        offenders.append(f"legacy namespace source must not be shipped: {relative}")

    for path in sorted(PACKAGE.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        relative = path.relative_to(ROOT)
        if "cstree.backtesting" in text:
            offenders.append(f"legacy namespace reference: {relative}")
        if "pkgutil import extend_path" in text:
            offenders.append(f"shared namespace mechanism: {relative}")

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
