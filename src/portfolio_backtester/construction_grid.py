from __future__ import annotations

from collections.abc import Callable
from typing import Any

from . import construction_grid_reports as _construction_grid_reports
from ._construction_grid_cli import add_construction_grid_args, run
from ._construction_grid_eval import build_construction_grid
from ._construction_grid_io import _read_returns_file

build_inertia_selection_report = _construction_grid_reports.build_inertia_selection_report
select_construction_variant_with_inertia = (
    _construction_grid_reports.select_construction_variant_with_inertia
)
write_reports = _construction_grid_reports.write_reports

BacktestTopKFn = Callable[..., Any]
DynamicEnsembleFn = Callable[..., tuple[Any, str, Any]]

__all__ = [
    "BacktestTopKFn",
    "DynamicEnsembleFn",
    "_read_returns_file",
    "add_construction_grid_args",
    "build_construction_grid",
    "build_inertia_selection_report",
    "run",
    "select_construction_variant_with_inertia",
    "write_reports",
]
