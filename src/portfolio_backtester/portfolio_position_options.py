"""Internal immutable option bundle for position construction."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .selection_controls import MaxNewNamesShortfallPolicy, controlled_selection_day


@dataclass(frozen=True)
class PortfolioPositionOptions:
    pred_col: str
    rebalance_dates: list[pd.Timestamp]
    shift_days: int
    top_k: int
    weighting_mode: str
    weighting_liquidity_col: str
    buffer_exit: int
    buffer_entry: int
    long_only: bool
    short_k: int | None
    group_col: str | None
    max_names_per_group: int | None
    liquidity_floor_col: str | None
    liquidity_floor_quantile: float | None
    max_turnover_per_rebalance: float | None
    rank_offset: int
    selection_tiebreak_col: str | None
    selection_score_bucket_size: float | None
    selection_score_margin: float | None
    selection_score_margin_col: str | None
    selection_score_margin_rank_limit: int | None
    selection_min_score: float | None
    max_new_names_per_rebalance: int | None
    max_new_names_shortfall_policy: MaxNewNamesShortfallPolicy
    max_positive_names: int | None

    def controlled_day(self, day: pd.DataFrame, *, ascending: bool) -> pd.DataFrame:
        """Return the side-specific weighting frame without duplicate controlled symbols."""

        return controlled_selection_day(
            day,
            self.pred_col,
            ascending=ascending,
            selection_tiebreak_col=self.selection_tiebreak_col,
            selection_score_bucket_size=self.selection_score_bucket_size,
            selection_min_score=self.selection_min_score,
            max_new_names_per_rebalance=self.max_new_names_per_rebalance,
        )
