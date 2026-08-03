from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def test_metric_correlation_helpers_remain_importable() -> None:
    from portfolio_backtester.metrics import pearson_corr, spearman_corr

    assert callable(pearson_corr)
    assert callable(spearman_corr)


def test_metric_facade_preserves_optional_scipy_hook(monkeypatch) -> None:
    import portfolio_backtester.metrics as metrics

    monkeypatch.setattr(metrics, "scipy_stats", None)

    summary = metrics.summarize_ic(pd.Series([0.1, -0.1, 0.2]))

    assert np.isnan(summary["p_value"])


def test_owner_native_layout() -> None:
    assert (SRC / "portfolio_backtester" / "__init__.py").is_file()


def test_namespace_boundary_ratchet() -> None:
    subprocess.run(
        [sys.executable, "scripts/dev/namespace_boundary.py"],
        cwd=ROOT,
        check=True,
    )
