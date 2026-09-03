"""Write position post-processing diagnostics to a run directory."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

_OUTPUTS = {
    "pre_repair_style": (
        "position_postprocess_pre_repair_style_path",
        "position_postprocess_pre_repair_style_exposure.csv",
    ),
    "pre_repair_industry": (
        "position_postprocess_pre_repair_industry_path",
        "position_postprocess_pre_repair_industry_exposure.csv",
    ),
    "pre_repair_active_summary": (
        "position_postprocess_pre_repair_active_summary_path",
        "position_postprocess_pre_repair_active_summary.csv",
    ),
    "breaches": (
        "position_postprocess_breaches_path",
        "position_postprocess_auto_breaches.csv",
    ),
}


def write_position_postprocess_outputs(
    *,
    ctx: Mapping[str, Any],
    run_dir: str | Path,
    artifacts: dict[str, Any],
) -> None:
    """Write non-empty position post-processing frames and update the artifact index."""
    frames = ctx.get("position_postprocess_artifacts")
    if not isinstance(frames, Mapping):
        return
    output_root = Path(run_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    for frame_key, (artifact_key, filename) in _OUTPUTS.items():
        frame = frames.get(frame_key)
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            continue
        path = output_root / filename
        frame.to_csv(path, index=False)
        artifacts[artifact_key] = path
