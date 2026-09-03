from pathlib import Path

import pandas as pd

from portfolio_backtester.position_outputs import write_position_outputs


def test_write_position_outputs_writes_latest_and_diff(tmp_path: Path) -> None:
    positions = pd.DataFrame(
        {
            "entry_date": ["20260102", "20260103", "20260103"],
            "symbol": ["600519.SH", "600519.SH", "000858.SZ"],
            "weight": [1.0, 0.5, 0.5],
        }
    )
    artifacts: dict[str, object] = {}

    write_position_outputs(
        positions=positions,
        run_dir=tmp_path,
        by_rebalance_name="positions.csv",
        current_name="current.csv",
        diff_name="diff.csv",
        artifacts=artifacts,
        by_rebalance_key="positions",
        current_key="current",
        diff_key="diff",
        enabled=True,
    )

    assert (tmp_path / "positions.csv").exists()
    assert pd.read_csv(tmp_path / "current.csv")["symbol"].tolist() == ["600519.SH", "000858.SZ"]
    diff = pd.read_csv(tmp_path / "diff.csv")
    assert set(diff["change"]) == {"changed", "added"}
    assert set(artifacts) == {"positions", "current", "diff"}
