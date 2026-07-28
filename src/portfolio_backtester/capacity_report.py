"""Capacity report (grid simulation + CLI).

Public symbols are re-exported here; their definitions live in the private
submodules :mod:`_capacity_report_config`, :mod:`_capacity_report_grid`, and
:mod:`_capacity_report_cli`. Original behavior is unchanged. ``capacity.py``
relies on ``from .capacity_report import *`` so all public names remain
reachable.
"""

from __future__ import annotations

from ._capacity_report_cli import add_capacity_report_args as add_capacity_report_args, run as run
from ._capacity_report_config import (
    DEFAULT_PARTICIPATION_RATES as DEFAULT_PARTICIPATION_RATES,
    DEFAULT_PORTFOLIO_VALUES as DEFAULT_PORTFOLIO_VALUES,
    DEFAULT_PRIMARY_PARTICIPATION_RATE as DEFAULT_PRIMARY_PARTICIPATION_RATE,
    THRESHOLD_PROFILES as THRESHOLD_PROFILES,
    CapacityThresholds as CapacityThresholds,
)
from ._capacity_report_grid import build_capacity_report as build_capacity_report

__all__ = [
    "DEFAULT_PARTICIPATION_RATES",
    "DEFAULT_PORTFOLIO_VALUES",
    "DEFAULT_PRIMARY_PARTICIPATION_RATE",
    "THRESHOLD_PROFILES",
    "CapacityThresholds",
    "add_capacity_report_args",
    "build_capacity_report",
    "run",
]
