"""Strict stateful and stateless OOS bridges for incumbent requalification."""

from __future__ import annotations

from typing import Literal, cast

import pandas as pd

from .daily_watch20_oos import (
    DEFAULT_SCORE_COLUMN,
    _numeric_series,
    _series,
    portfolio_daily_row,
)
from .incumbent_requalification import (
    IncumbentRequalificationConfig,
    IncumbentRequalificationPolicy,
    select_incumbent_requalified_portfolio,
)

SelectionStateMode = Literal["carry", "reset"]


def _required_columns(
    config: IncumbentRequalificationConfig,
    *,
    score_column: str,
) -> set[str]:
    return {
        config.date_col,
        config.symbol_col,
        config.industry_col,
        config.hard_eligibility_col,
        config.entry_eligibility_col,
        score_column,
        "forward_return_1d",
        "forward_label_start_date",
    }


def _validate_scored(
    scored: pd.DataFrame,
    config: IncumbentRequalificationConfig,
    *,
    score_column: str,
) -> None:
    if scored is None or scored.empty:
        raise ValueError("scored must be non-empty")
    missing = sorted(_required_columns(config, score_column=score_column) - set(scored.columns))
    if missing:
        raise ValueError(
            "incumbent OOS scored input missing required columns: " + ", ".join(missing)
        )
    if score_column != config.score_col and config.score_col in scored.columns:
        raise ValueError(
            f"score column collision: both {score_column!r} and {config.score_col!r} are present"
        )


def _canonical_selected(
    selected: pd.DataFrame,
    config: IncumbentRequalificationConfig,
) -> pd.DataFrame:
    rename: dict[str, str] = {}
    for source, target in (
        (config.date_col, "trade_date"),
        (config.symbol_col, "symbol"),
    ):
        if source == target:
            continue
        if target in selected.columns:
            raise ValueError(
                f"cannot map {source!r} to canonical {target!r}: target column already exists"
            )
        rename[source] = target
    return selected.rename(columns=rename)


def _incumbent_requalification_daily_rows(
    scored: pd.DataFrame,
    frame: pd.DataFrame,
    *,
    policy: IncumbentRequalificationPolicy,
    single_side_cost_bps: float,
    state_mode: SelectionStateMode,
    column_config: IncumbentRequalificationConfig | None = None,
    score_column: str = DEFAULT_SCORE_COLUMN,
) -> pd.DataFrame:
    config = column_config or IncumbentRequalificationConfig()
    if state_mode not in {"carry", "reset"}:
        raise ValueError("state_mode must be one of: carry, reset")
    _validate_scored(scored, config, score_column=score_column)

    rows: list[dict[str, object]] = []
    previous_weights: pd.Series | None = None
    previous_symbols: tuple[str, ...] | None = None
    incumbent_state: tuple[str, ...] = ()

    for _trade_date, date_rows in scored.groupby(config.date_col, sort=True):
        candidates = date_rows.copy()
        if score_column != config.score_col:
            candidates = candidates.rename(columns={score_column: config.score_col})

        result = select_incumbent_requalified_portfolio(
            candidates,
            previous_symbols=incumbent_state if state_mode == "carry" else (),
            policy=policy,
            config=config,
        )
        selected = result.positions.copy()
        receipt = result.receipt.to_dict()
        canonical = _canonical_selected(selected, config)
        symbols = tuple(_series(canonical, "symbol").astype(str))
        target = pd.Series(
            _numeric_series(canonical, "target_weight").to_numpy(dtype=float),
            index=pd.Index(symbols),
            dtype=float,
        )
        row = portfolio_daily_row(
            canonical,
            frame,
            target=target,
            expected_size=policy.portfolio_size,
            previous_weights=previous_weights,
            previous_symbols=previous_symbols,
            single_side_cost_bps=single_side_cost_bps,
        )
        row.update(
            {
                "retained_count": int(receipt.get("retained_count", 0)),
                "buffered_incumbent_count": int(receipt.get("buffered_incumbent_count", 0)),
                "new_position_count": int(receipt.get("new_position_count", 0)),
                "exited_count": int(receipt.get("exited_count", 0)),
                "cash_weight": float(receipt.get("cash_weight", 0.0)),
                "selected_symbols": "|".join(sorted(symbols)),
                "policy_id": policy.policy_id,
                "selection_state_mode": state_mode,
            }
        )
        rows.append(cast(dict[str, object], row))
        previous_weights = target
        previous_symbols = symbols
        incumbent_state = symbols if state_mode == "carry" else ()

    return pd.DataFrame(rows)


def stateful_incumbent_requalification_daily_rows(
    scored: pd.DataFrame,
    frame: pd.DataFrame,
    *,
    policy: IncumbentRequalificationPolicy,
    single_side_cost_bps: float,
    column_config: IncumbentRequalificationConfig | None = None,
    score_column: str = DEFAULT_SCORE_COLUMN,
) -> pd.DataFrame:
    """Run incumbent requalification while carrying prior holdings into selection."""

    return _incumbent_requalification_daily_rows(
        scored,
        frame,
        policy=policy,
        single_side_cost_bps=single_side_cost_bps,
        state_mode="carry",
        column_config=column_config,
        score_column=score_column,
    )


def stateless_incumbent_requalification_daily_rows(
    scored: pd.DataFrame,
    frame: pd.DataFrame,
    *,
    policy: IncumbentRequalificationPolicy,
    single_side_cost_bps: float,
    column_config: IncumbentRequalificationConfig | None = None,
    score_column: str = DEFAULT_SCORE_COLUMN,
) -> pd.DataFrame:
    """Reset selection state daily while preserving cross-date turnover accounting."""

    return _incumbent_requalification_daily_rows(
        scored,
        frame,
        policy=policy,
        single_side_cost_bps=single_side_cost_bps,
        state_mode="reset",
        column_config=column_config,
        score_column=score_column,
    )


__all__ = [
    "SelectionStateMode",
    "stateful_incumbent_requalification_daily_rows",
    "stateless_incumbent_requalification_daily_rows",
]
