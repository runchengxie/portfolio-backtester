from pathlib import Path

import pandas as pd

from portfolio_backtester.position_postprocess_outputs import write_position_postprocess_outputs


def test_write_position_postprocess_outputs_writes_non_empty_frames(tmp_path: Path) -> None:
    artifacts: dict[str, object] = {}
    write_position_postprocess_outputs(
        ctx={
            "position_postprocess_artifacts": {
                "pre_repair_style": pd.DataFrame({"factor": ["value"]}),
                "breaches": pd.DataFrame(),
            }
        },
        run_dir=tmp_path,
        artifacts=artifacts,
    )

    assert (tmp_path / "position_postprocess_pre_repair_style_exposure.csv").exists()
    assert not (tmp_path / "position_postprocess_auto_breaches.csv").exists()
    assert set(artifacts) == {"position_postprocess_pre_repair_style_path"}
