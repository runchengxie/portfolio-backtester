"""Portfolio-owned DailyWatch20 OOS construction and execution diagnostics."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd

from . import name_turnover, turnover_from_trade_weights
from .daily_watch20 import DailyWatch20Config as SelectionConfig
from .daily_watch20 import select_daily_watch20

DEFAULT_SCORE_COLUMN = "relative_percentile"


def _series(frame: pd.DataFrame, column: str) -> pd.Series:
    return cast(pd.Series, frame[column])


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return cast(pd.Series, pd.to_numeric(_series(frame, column), errors="coerce"))


def _flag_masks(values: pd.Series) -> tuple[pd.Series, pd.Series]:
    numeric = cast(pd.Series, pd.to_numeric(values, errors="coerce"))
    text = values.astype("string").str.strip().str.lower()
    true = numeric.eq(1).fillna(False) | text.isin(["true", "t", "yes", "y"])
    known = numeric.isin([0, 1]) | text.isin(
        ["true", "t", "yes", "y", "false", "f", "no", "n"]
    )
    return cast(pd.Series, true), cast(pd.Series, known)


def trade_audit(
    frame: pd.DataFrame,
    *,
    execution_date: pd.Timestamp,
    trade_weights: pd.Series,
) -> dict[str, float | bool | int]:
    """Audit direction-specific A-share tradability for proposed weight changes."""

    columns = ["symbol", "is_suspended", "open", "up_limit", "down_limit"]
    available = [column for column in columns if column in frame.columns]
    execution = frame.loc[
        _series(frame, "trade_date").eq(execution_date), available
    ].copy()
    execution = execution.drop_duplicates("symbol", keep="last").set_index("symbol")
    trades = trade_weights.rename("trade_weight").to_frame().join(execution, how="left")
    for column in columns[1:]:
        if column not in trades.columns:
            trades[column] = pd.NA
    suspended, suspended_known = _flag_masks(_series(trades, "is_suspended"))
    open_price = _numeric_series(trades, "open")
    up_limit = _numeric_series(trades, "up_limit")
    down_limit = _numeric_series(trades, "down_limit")
    open_known = open_price.notna() & np.isfinite(open_price) & open_price.gt(0)
    up_known = up_limit.notna() & np.isfinite(up_limit) & up_limit.gt(0)
    down_known = down_limit.notna() & np.isfinite(down_limit) & down_limit.gt(0)
    up_tolerance = pd.Series(
        np.maximum(up_limit.abs().to_numpy(dtype=float) * 1e-8, 1e-8),
        index=trades.index,
    )
    down_tolerance = pd.Series(
        np.maximum(down_limit.abs().to_numpy(dtype=float) * 1e-8, 1e-8),
        index=trades.index,
    )
    limit_up = open_known & up_known & open_price.ge(up_limit - up_tolerance)
    limit_down = open_known & down_known & open_price.le(down_limit + down_tolerance)
    trade_weight = _series(trades, "trade_weight")
    buys = trade_weight.gt(0)
    sells = trade_weight.lt(0)
    absolute = trade_weight.abs()
    buy_limit = buys & limit_up
    sell_limit = sells & limit_down
    suspended_trade = (buys | sells) & suspended
    unknown = (buys & ~(suspended_known & open_known & up_known)) | (
        sells & ~(suspended_known & open_known & down_known)
    )
    blocked = buy_limit | sell_limit | suspended_trade | unknown
    return {
        "buy_limit_blocked_weight": float(absolute.loc[buy_limit].sum()),
        "sell_limit_blocked_weight": float(absolute.loc[sell_limit].sum()),
        "suspended_trade_weight": float(absolute.loc[suspended_trade].sum()),
        "unknown_tradability_weight": float(absolute.loc[unknown].sum()),
        "blocked_gross_trade_weight": float(absolute.loc[blocked].sum()),
        "blocked_trade_count": int(blocked.sum()),
        "tradability_audit_passed": not bool(blocked.any()),
    }


def portfolio_daily_row(
    selected: pd.DataFrame,
    frame: pd.DataFrame,
    *,
    target: pd.Series,
    expected_size: int,
    previous_weights: pd.Series | None,
    previous_symbols: tuple[str, ...] | None,
    single_side_cost_bps: float,
) -> dict[str, Any]:
    symbols = tuple(_series(selected, "symbol").astype(str))
    trade_dates = pd.to_datetime(_series(selected, "trade_date")).dropna().unique()
    if len(trade_dates) != 1:
        raise ValueError("DailyWatch20 OOS selection must share one feature date")
    trade_date = cast(pd.Timestamp, pd.Timestamp(trade_dates[0]))
    prior = previous_weights if previous_weights is not None else pd.Series(dtype=float)
    all_symbols = prior.index.union(target.index)
    trades = target.reindex(all_symbols, fill_value=0.0) - prior.reindex(
        all_symbols, fill_value=0.0
    )
    turnover = turnover_from_trade_weights(trades, is_initial=previous_weights is None)
    execution_dates = (
        pd.to_datetime(_series(selected, "forward_label_start_date")).dropna().unique()
    )
    if len(execution_dates) != 1:
        raise ValueError("DailyWatch20 OOS rows must share one execution date")
    execution_date = cast(pd.Timestamp, pd.Timestamp(execution_dates[0]))
    audit = trade_audit(frame, execution_date=execution_date, trade_weights=trades)
    forward_return = _numeric_series(selected, "forward_return_1d")
    observed_return_count = int(forward_return.notna().sum())
    return_observation_complete = observed_return_count == len(selected)
    ordered_weight = target.reindex(pd.Index(symbols)).to_numpy(dtype=float)
    gross_return = (
        float(np.dot(forward_return.to_numpy(dtype=float), ordered_weight))
        if return_observation_complete
        else np.nan
    )
    transaction_cost = turnover.gross_traded_weight * single_side_cost_bps / 10_000.0
    net_return = (
        gross_return - transaction_cost if return_observation_complete else np.nan
    )
    return {
        "trade_date": trade_date,
        "execution_date": execution_date,
        "portfolio_size": len(selected),
        "portfolio_size_complete": len(selected) == expected_size,
        "observed_return_count": observed_return_count,
        "observed_return_ratio": observed_return_count / len(selected)
        if len(selected)
        else 0.0,
        "return_observation_complete": return_observation_complete,
        "gross_forward_return_proxy": gross_return,
        "transaction_cost": float(transaction_cost),
        "net_forward_return_proxy": net_return,
        "name_turnover": name_turnover(previous_symbols, symbols),
        **turnover.to_dict(),
        **audit,
    }


def portfolio_daily_rows(
    scored: pd.DataFrame,
    frame: pd.DataFrame,
    *,
    portfolio_size: int,
    single_side_cost_bps: float,
    score_column: str = DEFAULT_SCORE_COLUMN,
) -> pd.DataFrame:
    """Construct equal-weight top-N portfolios and deterministic daily diagnostics."""

    rows: list[dict[str, Any]] = []
    previous_weights: pd.Series | None = None
    previous_symbols: tuple[str, ...] | None = None
    for _trade_date, date_rows in scored.groupby("trade_date", sort=True):
        ranked = date_rows.sort_values(
            [score_column, "symbol"],
            ascending=[False, True],
            kind="mergesort",
        )
        selected = ranked.head(portfolio_size).copy()
        symbols = tuple(_series(selected, "symbol").astype(str))
        target_weight = 1.0 / len(selected) if len(selected) else 0.0
        target = pd.Series(target_weight, index=pd.Index(symbols), dtype=float)
        rows.append(
            portfolio_daily_row(
                selected,
                frame,
                target=target,
                expected_size=portfolio_size,
                previous_weights=previous_weights,
                previous_symbols=previous_symbols,
                single_side_cost_bps=single_side_cost_bps,
            )
        )
        previous_weights = target
        previous_symbols = symbols
    return pd.DataFrame(rows)


def guarded_a4b16_daily_rows(
    scored: pd.DataFrame,
    frame: pd.DataFrame,
    *,
    selection_config: SelectionConfig,
    single_side_cost_bps: float,
    score_column: str = DEFAULT_SCORE_COLUMN,
) -> pd.DataFrame:
    """Apply the stateful A4/B16 construction to rolling OOS scores."""

    rows: list[dict[str, Any]] = []
    previous_weights: pd.Series | None = None
    previous_symbols: tuple[str, ...] | None = None
    previous_b: tuple[str, ...] = ()
    for _trade_date, date_rows in scored.groupby("trade_date", sort=True):
        candidates = date_rows.rename(columns={score_column: "xgb_score"})
        result = select_daily_watch20(
            candidates,
            config=selection_config,
            previous_b_symbols=previous_b,
            fallback_mode="none",
        )
        selected = result.watchlist.copy()
        symbols = tuple(_series(selected, "symbol").astype(str))
        target = pd.Series(
            _numeric_series(selected, "tracking_weight").to_numpy(dtype=float),
            index=pd.Index(symbols),
            dtype=float,
        )
        row = portfolio_daily_row(
            selected,
            frame,
            target=target,
            expected_size=20,
            previous_weights=previous_weights,
            previous_symbols=previous_symbols,
            single_side_cost_bps=single_side_cost_bps,
        )
        sleeve = _series(selected, "sleeve")
        row.update(
            {
                "a_selected_count": int(sleeve.eq("A").sum()),
                "b_selected_count": int(sleeve.eq("B").sum()),
                "b_retained_count": int(_series(selected, "retained_b").sum()),
                "selected_symbols": "|".join(sorted(symbols)),
                "a_selected_symbols": "|".join(
                    sorted(_series(selected.loc[sleeve.eq("A")], "symbol").astype(str))
                ),
                "b_selected_symbols": "|".join(
                    sorted(_series(selected.loc[sleeve.eq("B")], "symbol").astype(str))
                ),
            }
        )
        rows.append(row)
        previous_weights = target
        previous_symbols = symbols
        previous_b = tuple(_series(selected.loc[sleeve.eq("B")], "symbol").astype(str))
    return pd.DataFrame(rows)


def portfolio_summary_fields(daily: pd.DataFrame) -> dict[str, Any]:
    """Summarize return completeness, turnover, costs and blocked trading."""

    observed = int(_series(daily, "observed_return_count").sum())
    expected = int(_series(daily, "portfolio_size").sum())
    complete = bool(_series(daily, "return_observation_complete").all())
    gross = _series(daily, "gross_forward_return_proxy")
    net = _series(daily, "net_forward_return_proxy")
    return {
        "gross_forward_return_proxy_mean": float(gross.mean()) if complete else np.nan,
        "net_forward_return_proxy_mean": float(net.mean()) if complete else np.nan,
        "gross_forward_return_proxy_sum": float(gross.sum()) if complete else np.nan,
        "net_forward_return_proxy_sum": float(net.sum()) if complete else np.nan,
        "transaction_cost_total": float(_series(daily, "transaction_cost").sum()),
        "mean_one_way_turnover": float(_series(daily, "one_way_turnover").mean()),
        "mean_name_turnover": float(_series(daily, "name_turnover").mean()),
        "mean_gross_traded_weight": float(_series(daily, "gross_traded_weight").mean()),
        "blocked_gross_trade_weight_total": float(
            _series(daily, "blocked_gross_trade_weight").sum()
        ),
        "buy_limit_blocked_weight_total": float(
            _series(daily, "buy_limit_blocked_weight").sum()
        ),
        "sell_limit_blocked_weight_total": float(
            _series(daily, "sell_limit_blocked_weight").sum()
        ),
        "suspended_trade_weight_total": float(
            _series(daily, "suspended_trade_weight").sum()
        ),
        "unknown_tradability_weight_total": float(
            _series(daily, "unknown_tradability_weight").sum()
        ),
        "tradability_audit_pass_ratio": float(
            _series(daily, "tradability_audit_passed").mean()
        ),
        "complete_portfolio_ratio": float(
            _series(daily, "portfolio_size_complete").mean()
        ),
        "observed_return_count": observed,
        "expected_return_count": expected,
        "observed_return_ratio": observed / expected if expected else 0.0,
        "complete_return_date_ratio": float(
            _series(daily, "return_observation_complete").mean()
        ),
    }


# Compatibility for the former strategy-pipeline private helper.
_portfolio_daily_rows = portfolio_daily_rows
_portfolio_daily_row = portfolio_daily_row
_trade_audit = trade_audit

__all__ = [
    "DEFAULT_SCORE_COLUMN",
    "_portfolio_daily_row",
    "_portfolio_daily_rows",
    "_trade_audit",
    "guarded_a4b16_daily_rows",
    "portfolio_daily_row",
    "portfolio_daily_rows",
    "portfolio_summary_fields",
    "trade_audit",
]
