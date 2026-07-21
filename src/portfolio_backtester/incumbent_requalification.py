"""Turnover-aware portfolio selection with separate entry and exit eligibility."""

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
_INTERNAL_COLUMNS = ("_score", "_hard_eligible", "_entry_eligible", "_full_rank")


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
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        digest = hashlib.sha256(encoded).hexdigest()[:16]
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
        values = asdict(self).values()
        if any(not str(value).strip() for value in values):
            raise ValueError("column names must be non-empty")
        if len(set(values)) != 6:
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
    """Selected positions and their construction receipt."""

    positions: pd.DataFrame
    receipt: IncumbentRequalificationReceipt


@dataclass(frozen=True, slots=True)
class _Prepared:
    frame: pd.DataFrame
    trade_date: str
    previous: frozenset[str]
    summary: Mapping[str, int]


def _truthy(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).ne(0)
    normalized = series.astype("string").str.strip().str.lower()
    return normalized.isin({"1", "true", "yes", "y"})


def _prepare(
    candidates: pd.DataFrame,
    previous_symbols: Collection[str],
    config: IncumbentRequalificationConfig,
) -> _Prepared:
    if candidates is None or candidates.empty:
        raise ValueError("candidates must be non-empty")
    required = set(asdict(config).values())
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(f"candidates missing required columns: {missing}")

    frame = candidates.copy()
    dates = frame[config.date_col].drop_duplicates()
    if len(dates) != 1 or bool(dates.isna().any()):
        raise ValueError("candidates must contain exactly one non-null trade date")
    parsed_date = pd.to_datetime(dates.iloc[0], errors="coerce")
    if pd.isna(parsed_date):
        raise ValueError(f"invalid trade date: {dates.iloc[0]!r}")
    trade_date = cast(pd.Timestamp, parsed_date).strftime("%Y-%m-%d")

    symbol = frame[config.symbol_col].astype("string").str.strip()
    if bool(symbol.isna().any()) or bool(symbol.eq("").any()):
        raise ValueError("symbols must be non-empty")
    if bool(symbol.duplicated().any()):
        raise ValueError("candidates must contain one row per symbol")
    frame[config.symbol_col] = symbol

    industry = frame[config.industry_col].astype("string").str.strip()
    if bool(industry.isna().any()) or bool(industry.eq("").any()):
        raise ValueError("industry values must be non-empty")
    frame[config.industry_col] = industry

    hard = _truthy(cast(pd.Series, frame[config.hard_eligibility_col]))
    entry = _truthy(cast(pd.Series, frame[config.entry_eligibility_col])) & hard
    score = pd.to_numeric(frame[config.score_col], errors="coerce")
    invalid = hard & (score.isna() | ~np.isfinite(score))
    if bool(invalid.any()):
        symbols = frame.loc[invalid, config.symbol_col].astype(str).tolist()[:5]
        raise ValueError(f"hard-eligible candidates require finite scores; invalid={symbols}")

    frame["_score"] = score
    frame["_hard_eligible"] = hard
    frame["_entry_eligible"] = entry
    eligible = frame.loc[hard].sort_values(
        ["_score", config.symbol_col], ascending=[False, True], kind="mergesort"
    )
    frame["_full_rank"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    frame.loc[eligible.index, "_full_rank"] = range(1, len(eligible) + 1)

    previous = frozenset(text for raw in previous_symbols if (text := str(raw).strip()))
    observable = previous & set(symbol.astype(str))
    return _Prepared(
        frame=frame,
        trade_date=trade_date,
        previous=previous,
        summary={
            "input_count": len(frame),
            "hard_eligible_count": int(hard.sum()),
            "entry_eligible_count": int(entry.sum()),
            "previous_count": len(previous),
            "previous_observable_count": len(observable),
        },
    )


def _ranked_symbols(frame: pd.DataFrame, config: IncumbentRequalificationConfig) -> list[str]:
    ranked = frame.sort_values(
        ["_score", config.symbol_col], ascending=[False, True], kind="mergesort"
    )
    return ranked[config.symbol_col].astype(str).tolist()


def _can_add(
    symbol: str,
    selected: Collection[str],
    rows: Mapping[str, pd.Series],
    config: IncumbentRequalificationConfig,
    policy: IncumbentRequalificationPolicy,
) -> bool:
    if symbol in selected:
        return False
    counts = Counter(str(rows[item][config.industry_col]) for item in selected)
    return counts[str(rows[symbol][config.industry_col])] < policy.industry_cap


def _replaceable_incumbents(
    selected: list[str],
    previous: frozenset[str],
    rank: Mapping[str, int],
    score: Mapping[str, float],
    policy: IncumbentRequalificationPolicy,
) -> list[str]:
    incumbents = [symbol for symbol in selected if symbol in previous]
    return sorted(
        incumbents,
        key=lambda symbol: (
            rank[symbol] <= policy.entry_rank_limit,
            score[symbol],
            symbol,
        ),
    )


def _select_symbols(
    eligible: pd.DataFrame,
    prepared: _Prepared,
    config: IncumbentRequalificationConfig,
    policy: IncumbentRequalificationPolicy,
) -> tuple[list[str], list[str], dict[str, int]]:
    rows = {str(row[config.symbol_col]): row for _, row in eligible.iterrows()}
    rank = {symbol: int(row["_full_rank"]) for symbol, row in rows.items()}
    score = {symbol: float(row["_score"]) for symbol, row in rows.items()}
    incumbent_frame = eligible.loc[
        eligible[config.symbol_col].astype(str).isin(prepared.previous)
        & eligible["_full_rank"].astype(int).le(policy.exit_rank_limit)
    ]
    entry_frame = eligible.loc[
        eligible["_entry_eligible"] & eligible["_full_rank"].astype(int).le(policy.entry_rank_limit)
    ]

    selected: list[str] = []
    blocked_incumbents = 0
    for symbol in _ranked_symbols(incumbent_frame, config):
        if len(selected) >= policy.portfolio_size:
            break
        if _can_add(symbol, selected, rows, config, policy):
            selected.append(symbol)
        else:
            blocked_incumbents += 1

    bootstrap = not prepared.previous
    budget = policy.portfolio_size if bootstrap else policy.max_new_positions
    opened: list[str] = []
    blocked_new = 0
    skipped_margin = 0
    entry_symbols = [
        symbol for symbol in _ranked_symbols(entry_frame, config) if symbol not in prepared.previous
    ]
    for incoming in entry_symbols:
        if len(opened) >= budget:
            break
        if len(selected) < policy.portfolio_size:
            if _can_add(incoming, selected, rows, config, policy):
                selected.append(incoming)
                opened.append(incoming)
            else:
                blocked_new += 1
            continue

        outgoing = next(
            (
                symbol
                for symbol in _replaceable_incumbents(
                    selected, prepared.previous, rank, score, policy
                )
                if _can_add(
                    incoming,
                    [item for item in selected if item != symbol],
                    rows,
                    config,
                    policy,
                )
            ),
            None,
        )
        if outgoing is None:
            blocked_new += 1
            continue
        if score[incoming] - score[outgoing] < policy.min_score_improvement:
            skipped_margin += 1
            continue
        selected[selected.index(outgoing)] = incoming
        opened.append(incoming)

    selected = sorted(selected, key=lambda symbol: (rank[symbol], symbol))
    return (
        selected,
        opened,
        {
            "bootstrap": int(bootstrap),
            "incumbent_exit_eligible_count": len(incumbent_frame),
            "entry_candidate_count": len(entry_frame),
            "industry_blocked_incumbent_count": blocked_incumbents,
            "industry_blocked_new_count": blocked_new,
            "score_margin_skipped_count": skipped_margin,
        },
    )


def select_incumbent_requalified_portfolio(
    candidates: pd.DataFrame,
    *,
    previous_symbols: Collection[str] = (),
    policy: IncumbentRequalificationPolicy | None = None,
    config: IncumbentRequalificationConfig | None = None,
) -> IncumbentRequalificationResult:
    """Select positions using strict entry and wider incumbent exit eligibility.

    New names require ``entry_eligible`` and an entry-buffer rank. Incumbents are
    re-scored on the current date and may remain through the wider exit buffer.
    A normal rebalance opens at most ``max_new_positions`` names. Missing slots
    remain cash at a fixed slot weight instead of levering the remaining names.
    """

    cfg = config or IncumbentRequalificationConfig()
    active = policy or IncumbentRequalificationPolicy()
    prepared = _prepare(candidates, previous_symbols, cfg)
    eligible = prepared.frame.loc[prepared.frame["_hard_eligible"]].copy()
    selected, opened, diagnostics = _select_symbols(eligible, prepared, cfg, active)
    if len(selected) < active.portfolio_size and not active.allow_cash:
        raise ValueError("portfolio could not be filled without cash under the frozen policy")

    selected_set = set(selected)
    retained = selected_set & prepared.previous
    positions = eligible.loc[eligible[cfg.symbol_col].astype(str).isin(selected_set)].copy()
    positions["full_rank"] = positions["_full_rank"].astype(int)
    positions["entry_eligible"] = positions["_entry_eligible"].astype(bool)
    symbols = positions[cfg.symbol_col].astype(str)
    positions["was_held"] = symbols.isin(prepared.previous)
    positions["retained"] = symbols.isin(retained)
    positions["new_position"] = symbols.isin(opened)
    positions["buffered_incumbent"] = positions["retained"] & positions["full_rank"].gt(
        active.entry_rank_limit
    )
    positions["target_weight"] = 1.0 / active.portfolio_size
    positions = (
        positions.sort_values(["full_rank", cfg.symbol_col], kind="mergesort")
        .drop(columns=list(_INTERNAL_COLUMNS))
        .reset_index(drop=True)
    )

    weight_sum = float(positions["target_weight"].sum()) if not positions.empty else 0.0
    industry_counts = {
        str(name): int(count)
        for name, count in positions[cfg.industry_col].value_counts().sort_index().items()
    }
    summary: dict[str, Any] = {
        **prepared.summary,
        "policy": active.to_dict(),
        **diagnostics,
        "bootstrap": bool(diagnostics["bootstrap"]),
        "selected_count": len(positions),
        "retained_count": len(retained),
        "buffered_incumbent_count": int(positions["buffered_incumbent"].sum()),
        "new_position_count": len(opened),
        "exited_count": len(prepared.previous - retained),
        "target_weight_sum": weight_sum,
        "cash_weight": max(0.0, 1.0 - weight_sum),
        "industry_counts": industry_counts,
        "unobservable_previous_count": len(prepared.previous)
        - int(prepared.summary["previous_observable_count"]),
    }
    receipt = IncumbentRequalificationReceipt(prepared.trade_date, active.policy_id, summary)
    return IncumbentRequalificationResult(positions, receipt)


__all__ = [
    "INCUMBENT_REQUALIFICATION_SCHEMA",
    "IncumbentRequalificationConfig",
    "IncumbentRequalificationPolicy",
    "IncumbentRequalificationReceipt",
    "IncumbentRequalificationResult",
    "select_incumbent_requalified_portfolio",
]
