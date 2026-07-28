"""DailyWatch20 construction: sleeve selection and watchlist assembly."""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Mapping, Sequence

import numpy as np
import pandas as pd

from ._daily_watch20_config import (
    DailyWatch20Config,
    DailyWatch20Receipt,
    DailyWatch20Result,
    FallbackMode,
    ReceiptStatus,
    _base_receipt_summary,
    _BSelection,
    _prepare_cross_section,
    _PreparedCrossSection,
    _raise_unavailable,
    _SleeveSelection,
    _validate_config,
)


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

    if len(selected) < count:
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
