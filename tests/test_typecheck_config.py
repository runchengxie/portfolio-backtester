from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_optional_qlib_unresolved_import_ignore_is_scoped_to_adapter() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    ty = config["tool"]["ty"]

    assert ty.get("rules", {}).get("unresolved-import") is None
    assert [
        override
        for override in ty.get("overrides", [])
        if override.get("rules", {}).get("unresolved-import") == "ignore"
    ] == [
        {
            "include": ["src/portfolio_backtester/backends/qlib.py"],
            "rules": {"unresolved-import": "ignore"},
        }
    ]
