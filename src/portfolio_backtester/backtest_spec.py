"""Serializable configuration for score-driven portfolio backtests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import pandas as pd

from .contracts import GroupCap, StrategySpec
from .execution import ExecutionModel, build_execution_model, describe_execution_model
from .selection_controls import (
    MaxNewNamesShortfallPolicy,
    SelectionPricePolicy,
    TargetWeightPolicy,
    validate_entry_rank_cutoff,
    validate_max_new_names_per_rebalance,
    validate_max_new_names_shortfall_policy,
    validate_max_positive_names,
    validate_selection_min_score,
    validate_selection_price_policy,
    validate_target_weight_policy,
)

BACKTEST_SPEC_SCHEMA_VERSION = 1

ExitMode = Literal["rebalance", "label_horizon"]


@dataclass(frozen=True)
class BacktestSpec:
    """Complete, data-free configuration for a score-driven backtest.

    Market frames stay outside the specification so the mapping returned by
    :meth:`to_mapping` can be stored as JSON or YAML. ``StrategySpec`` owns
    selection and allocation settings, while ``ExecutionModel`` remains the
    owner of execution, cost, slippage, exit, and entry constraints.
    """

    strategy: StrategySpec
    execution: ExecutionModel
    rebalance_dates: tuple[pd.Timestamp, ...]
    shift_days: int
    trading_days_per_year: int
    exit_mode: ExitMode = "rebalance"
    exit_horizon_days: int | None = None
    tradable_col: str | None = None
    liquidity_floor_col: str | None = None
    liquidity_floor_quantile: float | None = None
    weighting_liquidity_col: str = "medadv20_amount"
    max_turnover_per_rebalance: float | None = None
    rank_offset: int = 0
    selection_tiebreak_col: str | None = None
    selection_score_bucket_size: float | None = None
    selection_score_margin: float | None = None
    selection_score_margin_col: str | None = None
    selection_score_margin_rank_limit: int | None = None
    selection_min_score: float | None = None
    max_new_names_per_rebalance: int | None = None
    max_new_names_shortfall_policy: MaxNewNamesShortfallPolicy = "legacy_concentrate"
    max_positive_names: int | None = None
    entry_rank_cutoff: int | None = None
    selection_price_policy: SelectionPricePolicy = "execution_aware"
    target_weight_policy: TargetWeightPolicy = "normalized"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "selection_min_score",
            validate_selection_min_score(self.selection_min_score),
        )
        object.__setattr__(
            self,
            "max_new_names_shortfall_policy",
            validate_max_new_names_shortfall_policy(self.max_new_names_shortfall_policy),
        )
        object.__setattr__(
            self,
            "max_positive_names",
            validate_max_positive_names(self.max_positive_names),
        )
        object.__setattr__(
            self,
            "max_new_names_per_rebalance",
            validate_max_new_names_per_rebalance(self.max_new_names_per_rebalance),
        )
        object.__setattr__(
            self,
            "entry_rank_cutoff",
            validate_entry_rank_cutoff(self.entry_rank_cutoff),
        )
        object.__setattr__(
            self,
            "selection_price_policy",
            validate_selection_price_policy(self.selection_price_policy),
        )
        object.__setattr__(
            self,
            "target_weight_policy",
            validate_target_weight_policy(self.target_weight_policy),
        )
        if self.target_weight_policy == "fixed_slot" and self.strategy.weighting != "equal":
            raise ValueError("fixed_slot target weights require strategy.weighting='equal'.")
        if self.target_weight_policy == "fixed_slot" and not self.strategy.long_only:
            raise ValueError("fixed_slot target weights currently require a long-only strategy.")

    def to_mapping(self) -> dict[str, Any]:
        """Return a JSON/YAML-safe mapping without market data frames."""

        return {
            "schema_version": BACKTEST_SPEC_SCHEMA_VERSION,
            "strategy": self.strategy.to_dict(),
            "execution": _execution_to_mapping(self.execution),
            "rebalance_dates": [
                _normalized_date(date).strftime("%Y-%m-%d") for date in self.rebalance_dates
            ],
            "shift_days": self.shift_days,
            "trading_days_per_year": self.trading_days_per_year,
            "exit_mode": self.exit_mode,
            "exit_horizon_days": self.exit_horizon_days,
            "tradable_col": self.tradable_col,
            "liquidity_floor_col": self.liquidity_floor_col,
            "liquidity_floor_quantile": self.liquidity_floor_quantile,
            "weighting_liquidity_col": self.weighting_liquidity_col,
            "max_turnover_per_rebalance": self.max_turnover_per_rebalance,
            "rank_offset": self.rank_offset,
            "selection_tiebreak_col": self.selection_tiebreak_col,
            "selection_score_bucket_size": self.selection_score_bucket_size,
            "selection_score_margin": self.selection_score_margin,
            "selection_score_margin_col": self.selection_score_margin_col,
            "selection_score_margin_rank_limit": self.selection_score_margin_rank_limit,
            "selection_min_score": self.selection_min_score,
            "max_new_names_per_rebalance": self.max_new_names_per_rebalance,
            "max_new_names_shortfall_policy": self.max_new_names_shortfall_policy,
            "max_positive_names": self.max_positive_names,
            "entry_rank_cutoff": self.entry_rank_cutoff,
            "selection_price_policy": self.selection_price_policy,
            "target_weight_policy": self.target_weight_policy,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> BacktestSpec:
        """Build a specification from :meth:`to_mapping` output."""

        schema_version = int(value.get("schema_version", BACKTEST_SPEC_SCHEMA_VERSION))
        if schema_version != BACKTEST_SPEC_SCHEMA_VERSION:
            raise ValueError(f"Unsupported BacktestSpec schema version: {schema_version}")

        strategy = _strategy_from_mapping(_required_mapping(value, "strategy"))
        execution = _execution_from_mapping(_required_mapping(value, "execution"))
        return cls(
            strategy=strategy,
            execution=execution,
            rebalance_dates=_rebalance_dates_from_value(_required_value(value, "rebalance_dates")),
            shift_days=int(_required_value(value, "shift_days")),
            trading_days_per_year=int(_required_value(value, "trading_days_per_year")),
            exit_mode=cast(ExitMode, str(value.get("exit_mode", "rebalance"))),
            exit_horizon_days=_optional_int(value.get("exit_horizon_days")),
            tradable_col=_optional_str(value.get("tradable_col")),
            liquidity_floor_col=_optional_str(value.get("liquidity_floor_col")),
            liquidity_floor_quantile=_optional_float(value.get("liquidity_floor_quantile")),
            weighting_liquidity_col=str(value.get("weighting_liquidity_col", "medadv20_amount")),
            max_turnover_per_rebalance=_optional_float(value.get("max_turnover_per_rebalance")),
            rank_offset=int(value.get("rank_offset", 0)),
            selection_tiebreak_col=_optional_str(value.get("selection_tiebreak_col")),
            selection_score_bucket_size=_optional_float(value.get("selection_score_bucket_size")),
            selection_score_margin=_optional_float(value.get("selection_score_margin")),
            selection_score_margin_col=_optional_str(value.get("selection_score_margin_col")),
            selection_score_margin_rank_limit=_optional_int(
                value.get("selection_score_margin_rank_limit")
            ),
            selection_min_score=validate_selection_min_score(value.get("selection_min_score")),
            max_new_names_per_rebalance=validate_max_new_names_per_rebalance(
                value.get("max_new_names_per_rebalance")
            ),
            max_new_names_shortfall_policy=validate_max_new_names_shortfall_policy(
                value.get("max_new_names_shortfall_policy")
            ),
            max_positive_names=validate_max_positive_names(value.get("max_positive_names")),
            entry_rank_cutoff=validate_entry_rank_cutoff(value.get("entry_rank_cutoff")),
            selection_price_policy=validate_selection_price_policy(
                value.get("selection_price_policy")
            ),
            target_weight_policy=validate_target_weight_policy(value.get("target_weight_policy")),
        )


def _required_value(value: Mapping[str, Any], field: str) -> Any:
    if field not in value:
        raise ValueError(f"BacktestSpec mapping is missing required field: {field}")
    return value[field]


def _required_mapping(value: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    nested = _required_value(value, field)
    if not isinstance(nested, Mapping):
        raise ValueError(f"BacktestSpec {field} must be a mapping")
    return nested


def _strategy_from_mapping(value: Mapping[str, Any]) -> StrategySpec:
    group_cap_value = value.get("group_cap")
    group_cap = None
    if group_cap_value is not None:
        if not isinstance(group_cap_value, Mapping):
            raise ValueError("BacktestSpec strategy.group_cap must be a mapping")
        group_cap = GroupCap(
            column=str(_required_value(group_cap_value, "column")),
            max_names=int(_required_value(group_cap_value, "max_names")),
        )

    strategy_execution = value.get("execution")
    if strategy_execution is not None and not isinstance(strategy_execution, Mapping):
        raise ValueError("BacktestSpec strategy.execution must be a mapping")
    return StrategySpec(
        name=str(_required_value(value, "name")),
        type=str(_required_value(value, "type")),
        score_col=str(_required_value(value, "score_col")),
        top_k=int(_required_value(value, "top_k")),
        buffer_exit=int(value.get("buffer_exit", 0)),
        buffer_entry=int(value.get("buffer_entry", 0)),
        weighting=str(value.get("weighting", "equal")),
        long_only=bool(value.get("long_only", True)),
        short_k=_optional_int(value.get("short_k")),
        group_cap=group_cap,
        execution=dict(strategy_execution) if strategy_execution is not None else None,
        source=str(value.get("source", "explicit")),
    )


def _execution_from_mapping(value: Mapping[str, Any]) -> ExecutionModel:
    return build_execution_model(
        value,
        default_cost_bps=0.0,
        default_exit_price_policy="strict",
        default_exit_fallback_policy="ffill",
        default_price_col="close",
    )


def _execution_to_mapping(execution: ExecutionModel) -> dict[str, Any]:
    value = describe_execution_model(execution)
    try:
        restored = _execution_from_mapping(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "BacktestSpec serialization supports built-in execution models only"
        ) from exc
    if describe_execution_model(restored) != value:
        raise TypeError("BacktestSpec execution model does not have a stable mapping")
    return value


def _rebalance_dates_from_value(value: Any) -> tuple[pd.Timestamp, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("BacktestSpec rebalance_dates must be a sequence")
    dates: list[pd.Timestamp] = []
    for item in value:
        dates.append(_normalized_date(item))
    return tuple(dates)


def _normalized_date(value: Any) -> pd.Timestamp:
    date = pd.Timestamp(value)
    if pd.isna(date):
        raise ValueError("BacktestSpec rebalance_dates must contain valid dates")
    return cast(pd.Timestamp, date).normalize()


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


__all__ = ["BACKTEST_SPEC_SCHEMA_VERSION", "BacktestSpec"]
