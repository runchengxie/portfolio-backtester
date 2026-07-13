"""Construct the DailyWatch20 product watchlist from precomputed scores.

The module is intentionally model-agnostic.  It consumes one cross-section of
scores and applies portfolio-level constraints; alpha training and score
calculation remain outside ``portfolio-backtester``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

FallbackMode = Literal["none", "core20"]
ReceiptStatus = Literal["selected", "fallback", "unavailable"]

_OUTPUT_COLUMNS = {
    "blended_score",
    "dual_confirmed",
    "fallback_mode",
    "guard_prior",
    "ml_percentile",
    "retained_b",
    "selection_score",
    "sleeve",
    "sleeve_rank",
    "tracking_weight",
}


@dataclass(frozen=True)
class GuardFactorSpec:
    """One guard-prior component; factor values are ranked cross-sectionally."""

    column: str
    weight: float = 1.0
    higher_is_better: bool = True


@dataclass(frozen=True)
class DailyWatch20Config:
    """Construction settings for the strict four-name plus sixteen-name watchlist."""

    date_col: str = "trade_date"
    symbol_col: str = "symbol"
    industry_col: str = "first_industry_name"
    ml_score_col: str = "xgb_score"
    hard_eligibility_col: str = "hard_eligible"
    guard_factors: tuple[GuardFactorSpec, ...] = (GuardFactorSpec("guard_score"),)
    ml_weight: float = 0.60
    guard_weight: float = 0.40
    industry_cap: int = 4
    b_retention_buffer: int = 8
    b_max_replacements: int = 4
    a_tracking_weight: float = 0.20


@dataclass(frozen=True)
class DailyWatch20Receipt:
    """Auditable summary of one selection attempt."""

    status: ReceiptStatus
    trade_date: str | None
    fallback_mode: FallbackMode
    reason: str | None
    summary: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "trade_date": self.trade_date,
            "fallback_mode": self.fallback_mode,
            "reason": self.reason,
            **dict(self.summary),
        }


@dataclass(frozen=True)
class DailyWatch20Result:
    """Selected rows and the receipt that proves their construction."""

    watchlist: pd.DataFrame
    receipt: DailyWatch20Receipt


class DailyWatch20SelectionError(RuntimeError):
    """Fail-closed selection error carrying an unavailable receipt."""

    def __init__(self, message: str, receipt: DailyWatch20Receipt) -> None:
        super().__init__(message)
        self.receipt = receipt


@dataclass(frozen=True)
class _PreparedCrossSection:
    frame: pd.DataFrame
    trade_date: str
    input_summary: Mapping[str, int]


@dataclass(frozen=True)
class _BSelection:
    symbols: tuple[str, ...]
    retained: frozenset[str]
    previous_count: int
    exited_count: int
    added_count: int
    replacement_count: int
    forced_replacement_count: int


@dataclass(frozen=True)
class _SleeveSelection:
    a_symbols: tuple[str, ...]
    b_selection: _BSelection
    dual_confirmed: frozenset[str]


def _validate_config(config: DailyWatch20Config) -> None:
    text_fields = (
        config.date_col,
        config.symbol_col,
        config.industry_col,
        config.ml_score_col,
        config.hard_eligibility_col,
    )
    if any(not str(value).strip() for value in text_fields):
        raise ValueError("DailyWatch20 column names must be non-empty.")
    if not config.guard_factors:
        raise ValueError("DailyWatch20 requires at least one guard factor.")
    guard_columns = [factor.column for factor in config.guard_factors]
    if any(not str(column).strip() for column in guard_columns):
        raise ValueError("Guard factor column names must be non-empty.")
    if len(set(guard_columns)) != len(guard_columns):
        raise ValueError("Guard factor columns must be unique.")
    if any(not np.isfinite(factor.weight) or factor.weight <= 0 for factor in config.guard_factors):
        raise ValueError("Guard factor weights must be finite and positive.")
    if not np.isfinite(config.ml_weight) or not np.isfinite(config.guard_weight):
        raise ValueError("ML and guard weights must be finite.")
    if config.ml_weight < 0 or config.guard_weight < 0:
        raise ValueError("ML and guard weights must be non-negative.")
    if not np.isclose(config.ml_weight + config.guard_weight, 1.0):
        raise ValueError("ML and guard weights must sum to 1.")
    if config.industry_cap <= 0:
        raise ValueError("industry_cap must be positive.")
    if config.b_retention_buffer < 0 or config.b_max_replacements < 0:
        raise ValueError("B retention settings must be non-negative.")
    if not 0 <= config.a_tracking_weight <= 1:
        raise ValueError("a_tracking_weight must be between 0 and 1.")


def _required_columns(config: DailyWatch20Config) -> set[str]:
    return {
        config.date_col,
        config.symbol_col,
        config.industry_col,
        config.ml_score_col,
        config.hard_eligibility_col,
        *(factor.column for factor in config.guard_factors),
    }


def _truthy(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).ne(0)
    return series.astype("string").str.strip().str.lower().isin({"1", "true", "yes", "y"})


def _trade_date_text(value: object) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    if not pd.isna(parsed):
        return cast(pd.Timestamp, parsed).strftime("%Y-%m-%d")
    return str(value)


def _percentile_rank(values: pd.Series, *, higher_is_better: bool) -> pd.Series:
    return values.rank(
        method="average",
        ascending=higher_is_better,
        pct=True,
    )


def _prepare_cross_section(
    data: pd.DataFrame,
    config: DailyWatch20Config,
) -> _PreparedCrossSection:
    if data is None or data.empty:
        raise ValueError("DailyWatch20 input must be non-empty.")
    missing = sorted(_required_columns(config) - set(data.columns))
    if missing:
        raise ValueError(f"DailyWatch20 input missing required columns: {missing}")
    collisions = sorted(_OUTPUT_COLUMNS & set(data.columns))
    if collisions:
        raise ValueError(f"DailyWatch20 input contains reserved output columns: {collisions}")

    work = data.copy()
    dates = work[config.date_col].drop_duplicates()
    if len(dates) != 1 or bool(dates.isna().any()):
        raise ValueError("DailyWatch20 input must contain exactly one non-null trade date.")
    trade_date = _trade_date_text(dates.iloc[0])

    work[config.symbol_col] = work[config.symbol_col].astype("string").str.strip()
    symbols_valid = work[config.symbol_col].notna() & work[config.symbol_col].ne("")
    if not bool(symbols_valid.all()):
        raise ValueError("DailyWatch20 symbols must be non-empty.")
    if bool(work[config.symbol_col].duplicated().any()):
        raise ValueError("DailyWatch20 input must contain one row per symbol.")

    industry = work[config.industry_col].astype("string").str.strip()
    industry_valid = industry.notna() & industry.ne("")
    work[config.industry_col] = industry
    ml_score = pd.to_numeric(work[config.ml_score_col], errors="coerce")
    ml_valid = ml_score.notna() & np.isfinite(ml_score)
    hard_flag = _truthy(cast(pd.Series, work[config.hard_eligibility_col]))
    hard_eligible = hard_flag & industry_valid & ml_valid

    work["_ml_score"] = ml_score
    work["_hard_eligible"] = hard_eligible
    work["_ml_percentile"] = np.nan
    work.loc[hard_eligible, "_ml_percentile"] = _percentile_rank(
        ml_score.loc[hard_eligible], higher_is_better=True
    )

    guard_complete = hard_eligible.copy()
    numeric_guards: dict[str, pd.Series] = {}
    for factor in config.guard_factors:
        values = pd.to_numeric(work[factor.column], errors="coerce")
        numeric_guards[factor.column] = values
        guard_complete &= values.notna() & np.isfinite(values)

    factor_weight_sum = sum(factor.weight for factor in config.guard_factors)
    guard_prior = pd.Series(np.nan, index=work.index, dtype=float)
    if bool(guard_complete.any()):
        guard_prior.loc[guard_complete] = 0.0
        for factor in config.guard_factors:
            ranked = _percentile_rank(
                numeric_guards[factor.column].loc[guard_complete],
                higher_is_better=factor.higher_is_better,
            )
            guard_prior.loc[guard_complete] += ranked * factor.weight / factor_weight_sum

    work["_guard_prior"] = guard_prior
    work["_b_eligible"] = guard_complete
    work["_blended_score"] = (
        config.ml_weight * work["_ml_percentile"] + config.guard_weight * guard_prior
    )
    input_summary = {
        "input_rows": len(work),
        "hard_flag_true_count": int(hard_flag.sum()),
        "hard_eligible_count": int(hard_eligible.sum()),
        "b_eligible_count": int(guard_complete.sum()),
        "excluded_hard_flag_count": int((~hard_flag).sum()),
        "excluded_missing_ml_count": int((hard_flag & ~ml_valid).sum()),
        "excluded_missing_industry_count": int((hard_flag & ~industry_valid).sum()),
        "excluded_missing_guard_count": int((hard_eligible & ~guard_complete).sum()),
    }
    return _PreparedCrossSection(work, trade_date, input_summary)


def _ranked_symbols(frame: pd.DataFrame, config: DailyWatch20Config, score_col: str) -> list[str]:
    ranked = frame.sort_values(
        [score_col, "_ml_score", config.symbol_col],
        ascending=[False, False, True],
        kind="mergesort",
    )
    return ranked[config.symbol_col].astype(str).tolist()


def _industry_map(frame: pd.DataFrame, config: DailyWatch20Config) -> dict[str, str]:
    return dict(
        zip(
            frame[config.symbol_col].astype(str),
            frame[config.industry_col].astype(str),
            strict=False,
        )
    )


def _try_add_with_cap(
    symbol: str,
    *,
    selected: list[str],
    selected_set: set[str],
    industry_by_symbol: Mapping[str, str],
    industry_counts: Counter[str],
    industry_cap: int,
) -> bool:
    if symbol in selected_set:
        return False
    industry = industry_by_symbol[symbol]
    if industry_counts[industry] >= industry_cap:
        return False
    selected.append(symbol)
    selected_set.add(symbol)
    industry_counts[industry] += 1
    return True


def _select_ranked_with_cap(
    ranked_symbols: Sequence[str],
    *,
    count: int,
    industry_by_symbol: Mapping[str, str],
    industry_cap: int,
    initial_counts: Mapping[str, int] | None = None,
) -> tuple[list[str], Counter[str]]:
    selected: list[str] = []
    selected_set: set[str] = set()
    industry_counts = Counter(initial_counts or {})
    for symbol in ranked_symbols:
        _try_add_with_cap(
            symbol,
            selected=selected,
            selected_set=selected_set,
            industry_by_symbol=industry_by_symbol,
            industry_counts=industry_counts,
            industry_cap=industry_cap,
        )
        if len(selected) >= count:
            break
    return selected, industry_counts


def _add_previous_candidates(
    candidates: Sequence[str],
    *,
    target_retained: int,
    selected: list[str],
    selected_set: set[str],
    industry_by_symbol: Mapping[str, str],
    industry_counts: Counter[str],
    industry_cap: int,
) -> None:
    for symbol in candidates:
        _try_add_with_cap(
            symbol,
            selected=selected,
            selected_set=selected_set,
            industry_by_symbol=industry_by_symbol,
            industry_counts=industry_counts,
            industry_cap=industry_cap,
        )
        if len(selected) >= target_retained:
            break


def _select_b(
    frame: pd.DataFrame,
    *,
    count: int,
    previous_b_symbols: Collection[str],
    initial_industry_counts: Mapping[str, int],
    config: DailyWatch20Config,
) -> _BSelection:
    ranked = _ranked_symbols(frame, config, "_blended_score")
    industry_by_symbol = _industry_map(frame, config)
    rank_by_symbol = {symbol: rank for rank, symbol in enumerate(ranked, start=1)}
    previous = {str(symbol).strip() for symbol in previous_b_symbols if str(symbol).strip()}
    valid_previous = sorted(
        previous & set(ranked),
        key=lambda symbol: (rank_by_symbol[symbol], symbol),
    )
    buffer_limit = count + config.b_retention_buffer
    buffered_previous = [
        symbol for symbol in valid_previous if rank_by_symbol[symbol] <= buffer_limit
    ]
    comparable_previous_count = min(len(previous), count)
    minimum_retained = max(0, comparable_previous_count - config.b_max_replacements)

    selected: list[str] = []
    selected_set: set[str] = set()
    industry_counts = Counter(initial_industry_counts)
    _add_previous_candidates(
        buffered_previous,
        target_retained=count,
        selected=selected,
        selected_set=selected_set,
        industry_by_symbol=industry_by_symbol,
        industry_counts=industry_counts,
        industry_cap=config.industry_cap,
    )
    if len(selected) < minimum_retained:
        outside_buffer = [symbol for symbol in valid_previous if symbol not in selected_set]
        _add_previous_candidates(
            outside_buffer,
            target_retained=minimum_retained,
            selected=selected,
            selected_set=selected_set,
            industry_by_symbol=industry_by_symbol,
            industry_counts=industry_counts,
            industry_cap=config.industry_cap,
        )

    for symbol in ranked:
        _try_add_with_cap(
            symbol,
            selected=selected,
            selected_set=selected_set,
            industry_by_symbol=industry_by_symbol,
            industry_counts=industry_counts,
            industry_cap=config.industry_cap,
        )
        if len(selected) >= count:
            break

    retained = frozenset(selected_set & previous)
    exited_count = max(0, len(previous) - len(retained))
    added_count = max(0, len(selected_set - previous))
    replacement_count = max(0, comparable_previous_count - len(retained))
    forced_replacements = max(0, replacement_count - config.b_max_replacements)
    selected_in_score_order = sorted(selected, key=lambda symbol: (rank_by_symbol[symbol], symbol))
    return _BSelection(
        symbols=tuple(selected_in_score_order),
        retained=retained,
        previous_count=len(previous),
        exited_count=exited_count,
        added_count=added_count,
        replacement_count=replacement_count,
        forced_replacement_count=forced_replacements,
    )


def _base_receipt_summary(
    prepared: _PreparedCrossSection,
    config: DailyWatch20Config,
) -> dict[str, Any]:
    return {
        **dict(prepared.input_summary),
        "ml_weight": float(config.ml_weight),
        "guard_weight": float(config.guard_weight),
        "guard_factors": [
            {
                "column": factor.column,
                "weight": float(factor.weight),
                "higher_is_better": bool(factor.higher_is_better),
            }
            for factor in config.guard_factors
        ],
        "industry_cap": int(config.industry_cap),
        "b_retention_buffer": int(config.b_retention_buffer),
        "b_max_replacements": int(config.b_max_replacements),
    }


def _raise_unavailable(
    reason: str,
    *,
    prepared: _PreparedCrossSection,
    config: DailyWatch20Config,
    fallback_mode: FallbackMode,
    extra: Mapping[str, Any] | None = None,
) -> None:
    summary = _base_receipt_summary(prepared, config)
    summary.update(dict(extra or {}))
    receipt = DailyWatch20Receipt(
        status="unavailable",
        trade_date=prepared.trade_date,
        fallback_mode=fallback_mode,
        reason=reason,
        summary=summary,
    )
    raise DailyWatch20SelectionError(reason, receipt)


def _build_watchlist(
    prepared: _PreparedCrossSection,
    *,
    a_symbols: Sequence[str],
    b_selection: _BSelection,
    dual_confirmed: Collection[str],
    fallback_mode: FallbackMode,
    config: DailyWatch20Config,
) -> pd.DataFrame:
    work = prepared.frame.set_index(config.symbol_col, drop=False)
    b_symbols = list(b_selection.symbols)
    ordered_symbols = [*a_symbols, *b_symbols]
    selected = work.loc[ordered_symbols].copy().reset_index(drop=True)
    a_count = len(a_symbols)
    b_count = len(b_symbols)
    selected["sleeve"] = ["A"] * a_count + ["B"] * b_count
    selected["sleeve_rank"] = [*range(1, a_count + 1), *range(1, b_count + 1)]
    selected["ml_percentile"] = selected["_ml_percentile"].astype(float)
    selected["guard_prior"] = selected["_guard_prior"].astype(float)
    selected["blended_score"] = selected["_blended_score"].astype(float)
    selected["selection_score"] = np.where(
        selected["sleeve"].eq("A"),
        selected["ml_percentile"],
        selected["blended_score"],
    )
    selected["dual_confirmed"] = selected[config.symbol_col].isin(set(dual_confirmed))
    selected["retained_b"] = selected[config.symbol_col].isin(b_selection.retained) & selected[
        "sleeve"
    ].eq("B")
    if fallback_mode == "core20":
        selected["tracking_weight"] = 1.0 / 20.0
    else:
        selected["tracking_weight"] = np.where(
            selected["sleeve"].eq("A"),
            config.a_tracking_weight / 4.0,
            (1.0 - config.a_tracking_weight) / 16.0,
        )
    selected["fallback_mode"] = fallback_mode
    return selected.drop(
        columns=[
            "_ml_score",
            "_hard_eligible",
            "_ml_percentile",
            "_guard_prior",
            "_b_eligible",
            "_blended_score",
        ]
    )


def _validate_selected_watchlist(
    watchlist: pd.DataFrame,
    *,
    config: DailyWatch20Config,
    fallback_mode: FallbackMode,
) -> None:
    if len(watchlist) != 20 or watchlist[config.symbol_col].nunique() != 20:
        raise RuntimeError("DailyWatch20 invariant failed: expected exactly 20 unique symbols.")
    sleeve_counts = watchlist["sleeve"].value_counts().to_dict()
    expected = {"B": 20} if fallback_mode == "core20" else {"A": 4, "B": 16}
    if sleeve_counts != expected:
        raise RuntimeError(
            f"DailyWatch20 invariant failed: sleeve counts {sleeve_counts} != {expected}."
        )
    if not np.isclose(float(watchlist["tracking_weight"].sum()), 1.0):
        raise RuntimeError("DailyWatch20 invariant failed: tracking weights do not sum to 1.")
    industry_counts = watchlist[config.industry_col].value_counts()
    if not industry_counts.empty and int(industry_counts.max()) > config.industry_cap:
        raise RuntimeError("DailyWatch20 invariant failed: industry cap exceeded.")


def _select_sleeves(
    prepared: _PreparedCrossSection,
    *,
    config: DailyWatch20Config,
    previous_b_symbols: Collection[str],
    fallback_mode: FallbackMode,
) -> _SleeveSelection:
    work = prepared.frame
    industry_by_symbol = _industry_map(work, config)
    a_symbols: list[str] = []
    initial_industry_counts: Counter[str] = Counter()
    if fallback_mode == "none":
        a_candidates = work.loc[work["_hard_eligible"]]
        ranked_a = _ranked_symbols(a_candidates, config, "_ml_score")
        a_symbols = ranked_a[:4]
        if len(a_symbols) != 4:
            _raise_unavailable(
                "insufficient A candidates after hard eligibility",
                prepared=prepared,
                config=config,
                fallback_mode=fallback_mode,
                extra={"a_selected_count": len(a_symbols)},
            )
        initial_industry_counts = Counter(industry_by_symbol[symbol] for symbol in a_symbols)
        if initial_industry_counts and max(initial_industry_counts.values()) > config.industry_cap:
            _raise_unavailable(
                "pure-ML A selection exceeds the global industry cap",
                prepared=prepared,
                config=config,
                fallback_mode=fallback_mode,
                extra={
                    "a_selected_count": len(a_symbols),
                    "a_industry_counts": dict(initial_industry_counts),
                },
            )

    b_candidates_all = work.loc[work["_b_eligible"]]
    dual_reference_symbols, _ = _select_ranked_with_cap(
        _ranked_symbols(b_candidates_all, config, "_blended_score"),
        count=16,
        industry_by_symbol=industry_by_symbol,
        industry_cap=config.industry_cap,
    )
    b_candidates = b_candidates_all.loc[~b_candidates_all[config.symbol_col].isin(set(a_symbols))]
    b_count = 20 if fallback_mode == "core20" else 16
    b_selection = _select_b(
        b_candidates,
        count=b_count,
        previous_b_symbols=previous_b_symbols,
        initial_industry_counts=initial_industry_counts,
        config=config,
    )
    if len(b_selection.symbols) != b_count:
        _raise_unavailable(
            "insufficient B candidates after deduplication and industry cap",
            prepared=prepared,
            config=config,
            fallback_mode=fallback_mode,
            extra={
                "a_selected_count": len(a_symbols),
                "b_selected_count": len(b_selection.symbols),
            },
        )
    return _SleeveSelection(
        a_symbols=tuple(a_symbols),
        b_selection=b_selection,
        dual_confirmed=frozenset(set(a_symbols) & set(dual_reference_symbols)),
    )


def _build_success_receipt(
    prepared: _PreparedCrossSection,
    watchlist: pd.DataFrame,
    *,
    b_selection: _BSelection,
    config: DailyWatch20Config,
    fallback_mode: FallbackMode,
    fallback_reason: str | None,
) -> DailyWatch20Receipt:
    industry_counts = {
        str(industry): int(count)
        for industry, count in watchlist[config.industry_col].value_counts().sort_index().items()
    }
    summary = {
        **_base_receipt_summary(prepared, config),
        "selected_count": len(watchlist),
        "unique_symbol_count": int(watchlist[config.symbol_col].nunique()),
        "a_selected_count": int(watchlist["sleeve"].eq("A").sum()),
        "b_selected_count": int(watchlist["sleeve"].eq("B").sum()),
        "dual_confirmed_count": int(watchlist["dual_confirmed"].sum()),
        "tracking_weight_sum": float(watchlist["tracking_weight"].sum()),
        "industry_counts": industry_counts,
        "b_previous_count": b_selection.previous_count,
        "b_retained_count": len(b_selection.retained),
        "b_exited_count": b_selection.exited_count,
        "b_added_count": b_selection.added_count,
        "b_replacement_count": b_selection.replacement_count,
        "b_forced_replacement_count": b_selection.forced_replacement_count,
        "b_replacement_limit_forced": bool(b_selection.forced_replacement_count),
    }
    status: ReceiptStatus = "fallback" if fallback_mode == "core20" else "selected"
    return DailyWatch20Receipt(
        status=status,
        trade_date=prepared.trade_date,
        fallback_mode=fallback_mode,
        reason=(fallback_reason or "core20_requested") if fallback_mode == "core20" else None,
        summary=summary,
    )


def select_daily_watch20(
    data: pd.DataFrame,
    *,
    config: DailyWatch20Config | None = None,
    previous_b_symbols: Collection[str] = (),
    fallback_mode: FallbackMode = "none",
    fallback_reason: str | None = None,
) -> DailyWatch20Result:
    """Select one strict daily watchlist or raise with an unavailable receipt.

    ``fallback_mode="none"`` builds the normal A4+B16 layout.  A is ranked only
    by the ML score after hard eligibility.  ``fallback_mode="core20"`` is an
    explicit caller-controlled fallback that builds twenty B names; it is never
    activated silently.  Any attempt that cannot satisfy uniqueness, size, and
    industry constraints fails closed.
    """

    cfg = config or DailyWatch20Config()
    _validate_config(cfg)
    if fallback_mode not in {"none", "core20"}:
        raise ValueError("fallback_mode must be one of: none, core20.")
    prepared = _prepare_cross_section(data, cfg)
    selection = _select_sleeves(
        prepared,
        config=cfg,
        previous_b_symbols=previous_b_symbols,
        fallback_mode=fallback_mode,
    )
    watchlist = _build_watchlist(
        prepared,
        a_symbols=selection.a_symbols,
        b_selection=selection.b_selection,
        dual_confirmed=selection.dual_confirmed,
        fallback_mode=fallback_mode,
        config=cfg,
    )
    _validate_selected_watchlist(watchlist, config=cfg, fallback_mode=fallback_mode)
    receipt = _build_success_receipt(
        prepared,
        watchlist,
        b_selection=selection.b_selection,
        config=cfg,
        fallback_mode=fallback_mode,
        fallback_reason=fallback_reason,
    )
    return DailyWatch20Result(watchlist=watchlist, receipt=receipt)


__all__ = [
    "DailyWatch20Config",
    "DailyWatch20Receipt",
    "DailyWatch20Result",
    "DailyWatch20SelectionError",
    "GuardFactorSpec",
    "select_daily_watch20",
]
