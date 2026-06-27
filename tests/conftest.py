from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent

sys.path[:0] = [
    str(ROOT / "src"),
    str(WORKSPACE / "cross-sectional-trees" / "src"),
    str(WORKSPACE / "alpha-research" / "src"),
]
