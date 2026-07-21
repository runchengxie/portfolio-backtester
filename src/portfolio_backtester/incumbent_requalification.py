"""Turnover-aware portfolio selection with separate entry and exit eligibility.

The selector is model-agnostic. Callers provide one scored cross-section and the
previous portfolio. New positions must satisfy the stricter entry rule, while
incumbents may remain through a wider exit buffer after being re-scored on the
current date. When the replacement budget cannot restore a full portfolio, the
unallocated slots remain cash instead of being redistributed.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Collection, Mapping
from dataclasses import asdict, dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

INCUMBENT_REQUALIFICATION_SCHEMA = "portfolio_backtester.incumbent_requalification.v1"


@dataclass(frozen=True, slots=True)
class IncumbentRequalificationPolicy:
    """Frozen construction policy for turnover-aware incumbent requalification."""

    portfolio_size: int = 20
    entry_rank_limit: int = 20
    exit_rank_limit: int = 40
    max_new_positions: int = 4
    industry_cap: int = 4
    min_score_improvement: float = 0.0
    allow_cash: bool = True

    def __post_init__(self) -> None:
        if self.portfolio_size <= 0:
            raise ValueError("portfolio_size must be positive")
        if self.entry_rank_limit <= 0:
            raise ValueError("entry_rank_limit must be positive")
        if self.exit_rank_limit < self.entry_rank_limit:
            raise ValueError("exit_rank_limit must be >= entry_rank_limit")
        if not 0 <= self.max_new_positions <= self.portfolio_size:
            raise ValueError("max_new_positions must be in [0, portfolio_size]")
        if self.industry_cap <= 0:
            raise ValueError("industry_cap must be positive")
        if not np.isfinite(self.min_score_improvement) or self.min_score_improvement < 0:
            raise ValueError("min_score_improvement must be finite and non-negative")

    @property
    def policy_id(self) -> str:
        payload = {"schema_version": INCUMBENT_REQUALIFICATION_SCHEMA, **asdict(self)}
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16]
        return f"{INCUMBENT_REQUALIFICATION_SCHEMA}:{digest}"

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": INCUMBENT_REQUALIFICATION_SCHEMA,
            "policy_id": self.policy_id,
            **asdict(self),
        }


@dataclass(frozen=True, slots=True)
class IncumbentRequalificationConfig:
    """Column mapping for :func:`select_incumbent_requalified_portfolio`."""

    date_col: str = "trade_date"
    symbol_col: str = "symbol"
    score_col: str = "selection_score"
    industry_col: str = "industry"
    hard_eligibility_col: str = "hard_eligible"
    entry_eligibility_col: str = "entry_eligible"

    def __post_init__(self) -> None:
        values = (
            self.date_col,
            self.symbol_col,
            self.score_col,
            self.industry_col,
            self.hard_eligibility_col,
            self.entry_eligibility_col,
        )
        if any(not str(value).strip() for value in values):
            raise ValueError("column names must be non-empty")
        if len(set(values)) != len(values):
            raise ValueError("column names must be unique")


@dataclass(frozen=True, slots=True)
class IncumbentRequalificationReceipt:
    """Auditable summary of one portfolio construction decision."""

    trade_date: str
    policy_id: str
    summary: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": INCUMBENT_REQUALIFICATION_SCHEMA,
            "trade_date": self.trade_date,
            "policy_id": self.policy_id,
            **dict(self.summary),
        }


@dataclass(frozen=True, slots=True)
class IncumbentRequalificationResult:
    """Selected positions and the receipt proving how they were constructed."""

    positions: pd.DataFrame
    receipt: IncumbentRequalificationReceipt


@dataclass(frozen=True, slots=True)
class _PreparedCrossSection:
    frame: pd.DataFrame
    trade_date: str
    previous_symbols: frozenset[str]
    input_summary: Mapping[str, int]


def _truthy(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).ne(0)
    return series.astype("string").str.strip().str.lower().isin({"1", "true", "yes", "y"})


def _trade_date_text(value: object) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"invalid trade date: {value!r}")
    return cast(pd.Timestamp, parsed).strftime("%Y-%m-%d")


def _prepare_cross_section(
    candidates: pd.DataFrame,
    previous_symbols: Collection[str],
    config: IncumbentRequalificationConfig,
) -> _PreparedCrossSection:
    if candidates is None or candidates.empty:
        raise ValueError("candidates must be non-empty")
    required = {
        config.date_col,
        config.symbol_col,
        config.score_col,
        config.industry_col,
        config.hard_eligibility_col,
        config.entry_eligibility_col,
    }
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(f"candidates missing required columns: {missing}")

    frame = candidates.copy()
    dates = frame[config.date_col].drop_duplicates()
    if len(dates) != 1 or bool(dates.isna().any()):
        raise ValueError("candidates must contain exactly one non-null trade date")
    trade_date = _trade_date_text(dates.iloc[0])

    frame[config.symbol_col] = frame[config.symbol_col].astype("string").str.strip()
    if bool(frame[config.symbol_col].isna().any()) or bool(frame[config.symbol_col].eq("").any()):
        raise ValueError("symbols must be non-empty")
    if bool(frame[config.symbol_col].duplicated().any()):
        raise ValueError("candidates must contain one row per symbol")

    frame[config.industry_col] = frame[config.industry_col].astype("string").str.strip()
    if bool(frame[config.industry_col].isna().any()) or bool(
        frame[config.industry_col].eq("").any()
    ):
        raise ValueError("industry values must be non-empty")

    hard_eligible = _truthy(cast(pd.Series, frame[config.hard_eligibility_col]))
    entry_eligible = _truthy(cast(pd.Series, frame[config.entry_eligibility_col])) & hard_eligible
    score = pd.to_numeric(frame[config.score_col], errors="coerce")
    invalid_hard_score = hard_eligible & (score.isna() | ~np.isfinite(score))
    if bool(invalid_hard_score.any()):
        invalid = frame.loc[invalid_hard_score, config.symbol_col].astype(str).tolist()[:5]
        raise ValueError(
            "hard-eligible candidates require finite scores; "
            f"invalid symbols={invalid}"
        )

    frame["_score"] = score
    frame["_hard_eligible"] = hard_eligible
    frame["_entry_eligible"] = entry_eligible
    eligible = frame.loc[hard_eligible].sort_values(
        ["_score", config.symbol_col], ascending=[False, True], kind="mergesort"
    )
    rank_by_index = pd.Series(range(1, len(eligible) + 1), index=eligible.index, dtype="int64")
    frame["_full_rank"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    frame.loc[eligible.index, "_full_rank"] = rank_by_index

    previous = frozenset(text for raw in previous_symbols if (text := str(raw).strip()))
    observable_previous = previous & set(frame[config.symbol_col].astype(str))
    return _PreparedCrossSection(
        frame=frame,
        trade_date=trade_date,
        previous_symbols=previous,
        input_summary={
            "input_count": len(frame),
            "hard_eligible_count": int(hard_eligible.sum()),
            "entry_eligible_count": int(entry_eligible.sum()),
            "previous_count": len(previous),
            "previous_observable_count": len(observable_previous),
        },
    )


def _industry_counts(
    selected: list[str],
    row_by_symbol: Mapping[str, pd.Series],
    industry_col: str,
) -> Counter[str]:
    return Counter(str(row_by_symbol[symbol][industry_col]) for symbol in selected)


def _can_add(
    symbol: str,
    *,
    selected: list[str],
    row_by_symbol: Mapping[str, pd.Series],
    industry_col: str,
    industry_cap: int,
) -> bool:
    if symbol in selected:
        return False
    counts = _industry_counts(selected, row_by_symbol, industry_col)
    industry = str(row_by_symbol[symbol][industry_col])
    return counts[industry] < industry_cap


def _can_replace(
    outgoing: str,
    incoming: str,
    *,
    selected: list[str],
    row_by_symbol: Mapping[str, pd.Series],
    industry_col: str,
    industry_cap: int,
) -> bool:
    remaining = [symbol for symbol in selected if symbol != outgoing]
    return _can_add(
        incoming,
        selected=remaining,
        row_by_symbol=row_by_symbol,
        industry_col=industry_col,
        industry_cap=industry_cap,
    )


def _ranked_symbols(frame: pd.DataFrame, config: IncumbentRequalificationConfig) -> list[str]:
    ranked = frame.sort_values(
        ["_score", config.symbol_col], ascending=[False, True], kind="mergesort"
    )
    return ranked[config.symbol_col].astype(str).tolist()


def select_incumbent_requalified_portfolio(
    candidates: pd.DataFrame,
    *,
    previous_symbols: Collection[str] = (),
    policy: IncumbentRequalificationPolicy | None = None,
    config: IncumbentRequalificationConfig | None = None,
) -> IncumbentRequalificationResult:
    """Build a turnover-aware portfolio from one scored cross-section.

    Entry eligibility is deliberately stricter than incumbent eligibility:

    * New names must be entry-eligible and rank within ``entry_rank_limit``.
    * Existing names may remain while hard-eligible and ranked within
      ``exit_rank_limit``.
    * No more than ``max_new_positions`` names are opened on a normal rebalance.
      The first portfolio is allowed to bootstrap to ``portfolio_size``.
    * Unfilled slots remain cash. Selected names keep one slot of weight each,
      so retaining fewer names does not silently lever the remaining holdings.
    """

    cfg = config or IncumbentRequalificationConfig()
    active_policy = policy or IncumbentRequalificationPolicy()
    prepared = _prepare_cross_section(candidates, previous_symbols, cfg)
    frame = prepared.frame
    eligible = frame.loc[frame["_hard_eligible"]].copy()
    row_by_symbol = {str(row[cfg.symbol_col]): row for _, row in eligible.iterrows()}
    rank_by_symbol = {
        str(row[cfg.symbol_col]): int(row["_full_rank"])
        for _, row in eligible.iterrows()
    }
    score_by_symbol = {
        str(row[cfg.symbol_col]): float(row["_score"])
        for _, row in eligible.iterrows()
    }

    incumbent_frame = eligible.loc[
        eligible[cfg.symbol_col].astype(str).isin(prepared.previous_symbols)
        & eligible["_full_rank"].astype(int).le(active_policy.exit_rank_limit)
    ]
    incumbent_symbols = _ranked_symbols(incumbent_frame, cfg)
    entry_frame = eligible.loc[
        eligible["_entry_eligible"]
        & eligible["_full_rank"].astype(int).le(active_policy.entry_rank_limit)
    ]
    new_entry_symbols = [
        symbol
        for symbol in _ranked_symbols(entry_frame, cfg)
        if symbol not in prepared.previous_symbols
    ]

    selected: list[str] = []
    blocked_incumbents: list[str] = []
    for symbol in incumbent_symbols:
        if len(selected) >= active_policy.portfolio_size:
            break
        if _can_add(
            symbol,
            selected=selected,
            row_by_symbol=row_by_symbol,
            industry_col=cfg.industry_col,
            industry_cap=active_policy.industry_cap,
        ):
            selected.append(symbol)
        else:
            blocked_incumbents.append(symbol)

    bootstrap = not prepared.previous_symbols
    new_budget = active_policy.portfolio_size if bootstrap else active_policy.max_new_positions
    opened: list[str] = []
    skipped_margin: list[str] = []
    blocked_new: list[str] = []
    for incoming in new_entry_symbols:
        if len(opened) >= new_budget:
            break
        if len(selected) < active_policy.portfolio_size:
            if _can_add(
                incoming,
                selected=selected,
                row_by_symbol=row_by_symbol,
                industry_col=cfg.industry_col,
                industry_cap=active_policy.industry_cap,
            ):
                selected.append(incoming)
                opened.append(incoming)
            else:
                blocked_new.append(incoming)
            continue

        replaceable = [symbol for symbol in selected if symbol in prepared.previous_symbols]
        replaceable.sort(
            key=lambda symbol: (
                rank_by_symbol[symbol] <= active_policy.entry_rank_limit,
                score_by_symbol[symbol],
                symbol,
            )
        )
        outgoing = next(
            (
                symbol
                for symbol in replaceable
                if _can_replace(
                    symbol,
                    incoming,
                    selected=selected,
                    row_by_symbol=row_by_symbol,
                    industry_col=cfg.industry_col,
                    industry_cap=active_policy.industry_cap,
                )
            ),
            None,
        )
        if outgoing is None:
            blocked_new.append(incoming)
            continue
        improvement = score_by_symbol[incoming] - score_by_symbol[outgoing]
        if improvement < active_policy.min_score_improvement:
            skipped_margin.append(incoming)
            continue
        selected[selected.index(outgoing)] = incoming
        opened.append(incoming)

    selected = sorted(selected, key=lambda symbol: (rank_by_symbol[symbol], symbol))
    selected_set = set(selected)
    retained = selected_set & prepared.previous_symbols
    exited = prepared.previous_symbols - retained
    if len(selected) < active_policy.portfolio_size and not active_policy.allow_cash:
        raise ValueError(
            "portfolio could not be filled without cash under the frozen "
            "eligibility and turnover policy"
        )

    positions = eligible.loc[eligible[cfg.symbol_col].astype(str).isin(selected_set)].copy()
    positions["full_rank"] = positions["_full_rank"].astype(int)
    positions["entry_eligible"] = positions["_entry_eligible"].astype(bool)
    symbols = positions[cfg.symbol_col].astype(str)
    positions["was_held"] = symbols.isin(prepared.previous_symbols)
    positions["retained"] = symbols.isin(retained)
    positions["new_position"] = symbols.isin(opened)
    positions["buffered_incumbent"] = positions["retained"] & positions["full_rank"].gt(
        active_policy.entry_rank_limit
    )
    positions["target_weight"] = 1.0 / active_policy.portfolio_size
    positions = positions.sort_values(
        ["full_rank", cfg.symbol_col], ascending=[True, True], kind="mergesort"
    ).drop(columns=["_score", "_hard_eligible", "_entry_eligible", "_full_rank"])
    positions = positions.reset_index(drop=True)

    target_weight_sum = float(positions["target_weight"].sum()) if not positions.empty else 0.0
    cash_weight = max(0.0, 1.0 - target_weight_sum)
    industry_counts = {
        str(name): int(count)
        for name, count in positions[cfg.industry_col].value_counts().sort_index().items()
    }
    summary: dict[str, Any] = {
        **dict(prepared.input_summary),
        "policy": active_policy.to_dict(),
        "bootstrap": bootstrap,
        "incumbent_exit_eligible_count": len(incumbent_symbols),
        "entry_candidate_count": len(entry_frame),
        "selected_count": len(positions),
        "retained_count": len(retained),
        "buffered_incumbent_count": int(positions["buffered_incumbent"].sum()),
        "new_position_count": len(opened),
        "exited_count": len(exited),
        "target_weight_sum": target_weight_sum,
        "cash_weight": cash_weight,
        "industry_counts": industry_counts,
        "industry_blocked_incumbent_count": len(blocked_incumbents),
        "industry_blocked_new_count": len(blocked_new),
        "score_margin_skipped_count": len(skipped_margin),
        "unobservable_previous_count": len(prepared.previous_symbols)
        - int(dict(prepared.input_summary)["previous_observable_count"]),
    }
    receipt = IncumbentRequalificationReceipt(
        trade_date=prepared.trade_date,
        policy_id=active_policy.policy_id,
        summary=summary,
    )
    return IncumbentRequalificationResult(positions=positions, receipt=receipt)


__all__ = [
    "INCUMBENT_REQUALIFICATION_SCHEMA",
    "IncumbentRequalificationConfig",
    "IncumbentRequalificationPolicy",
    "IncumbentRequalificationReceipt",
    "IncumbentRequalificationResult",
    "select_incumbent_requalified_portfolio",
]
