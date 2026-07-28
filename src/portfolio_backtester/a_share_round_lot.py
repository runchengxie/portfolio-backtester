"""A-share round-lot target selection and allocation.

Public helpers are re-exported here; their definitions live in the private
submodules :mod:`_round_lot_weights`, :mod:`_round_lot_targets`, and
:mod:`_round_lot_allocate`. Original behavior is unchanged.
"""

from __future__ import annotations

from ._round_lot_allocate import (
    allocate_round_lot as allocate_round_lot,
    allocate_round_lot_account as allocate_round_lot_account,
    portfolio_value as portfolio_value,
)
from ._round_lot_targets import select_round_lot_targets as select_round_lot_targets
from ._round_lot_weights import (
    RoundLotVariant as RoundLotVariant,
    cap_and_redistribute as cap_and_redistribute,
)

__all__ = [
    "RoundLotVariant",
    "allocate_round_lot",
    "allocate_round_lot_account",
    "cap_and_redistribute",
    "portfolio_value",
    "select_round_lot_targets",
]
