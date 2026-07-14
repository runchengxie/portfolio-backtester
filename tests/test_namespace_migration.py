from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def test_owner_native_layout() -> None:
    assert (SRC / "portfolio_backtester" / "__init__.py").is_file()


def test_namespace_boundary_ratchet() -> None:
    subprocess.run(
        [sys.executable, "scripts/dev/namespace_boundary.py"],
        cwd=ROOT,
        check=True,
    )
