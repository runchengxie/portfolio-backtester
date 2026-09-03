from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import yaml

from portfolio_backtester.afml_evidence import (
    generate_run_afml_evidence,
    merge_evidence_fragment,
)


def _write_run(root: Path) -> None:
    dates = pd.date_range("2023-01-06", periods=20, freq="7D")
    returns = np.tile([0.01, 0.008, -0.006, 0.004], 5)
    for name, column, values in (
        ("backtest_net.csv", "net_return", returns),
        ("backtest_gross.csv", "gross_return", returns + 0.001),
        ("backtest_turnover.csv", "turnover", np.full(20, 0.2)),
    ):
        pd.DataFrame({"period_end": dates, column: values}).to_csv(root / name, index=False)
    pd.DataFrame({"symbol": ["A", "B"], "weight": [0.6, 0.4]}).to_csv(
        root / "positions_current.csv", index=False
    )
    (root / "config.used.yml").write_text(
        yaml.safe_dump({"strategy": {"weighting": "equal"}, "backtest": {"top_k": 2}}),
        encoding="utf-8",
    )


def test_generate_run_afml_evidence_writes_sidecars(tmp_path: Path) -> None:
    _write_run(tmp_path)
    fragment = generate_run_afml_evidence(tmp_path, bootstrap_samples=20, random_state=7)

    assert (tmp_path / "sizing_receipt.json").is_file()
    assert (tmp_path / "strategy_risk_report.json").is_file()
    assert (tmp_path / "afml_evidence_fragment.json").is_file()
    evidence = cast(dict[str, Any], fragment["evidence"])
    assert cast(dict[str, Any], evidence["sizing_receipt"])["status"] == "pass"


def test_generated_evidence_merges_into_manifest(tmp_path: Path) -> None:
    _write_run(tmp_path)
    fragment = generate_run_afml_evidence(tmp_path, bootstrap_samples=20, random_state=3)

    merged = merge_evidence_fragment({"evidence": {}}, fragment)

    evidence = cast(dict[str, Any], merged["evidence"])
    assert cast(dict[str, Any], evidence["strategy_risk"])["status"] == "pass"
