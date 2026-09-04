"""Owner-native orchestration for saved-position share allocation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .allocation_reference import (
    AllocationReference,
    join_allocation_reference,
    load_allocation_reference,
)
from .allocation_selection import prepare_selection
from .share_allocation import allocate_equal_weight_shares


@dataclass(frozen=True, slots=True)
class EqualWeightAllocation:
    """Allocation result and the reference snapshot used to produce it."""

    allocations: pd.DataFrame
    reference: AllocationReference
    investable_cash: float
    estimated_value: float
    cash_left: float


def build_equal_weight_allocation(
    selection: pd.DataFrame,
    *,
    reference_file: str | Path,
    side: str = "long",
    top_n: int = 20,
    cash: float = 1_000_000,
    buffer_bps: float = 0.0,
) -> EqualWeightAllocation:
    """Prepare holdings and allocate round-lot shares from one reference snapshot."""

    prepared = prepare_selection(selection, side=side, top_n=top_n)
    reference = load_allocation_reference(reference_file)
    referenced = join_allocation_reference(prepared, reference)
    result = allocate_equal_weight_shares(
        referenced,
        cash=cash,
        buffer_bps=buffer_bps,
    )
    return EqualWeightAllocation(
        allocations=result.allocations,
        reference=reference,
        investable_cash=result.investable_cash,
        estimated_value=result.est_total,
        cash_left=result.cash_left,
    )


__all__ = ["EqualWeightAllocation", "build_equal_weight_allocation"]
