from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from .contracts import assert_positions_by_rebalance_frame
from .engine import _compute_trade_summary
from .execution import BpsCostModel, ExitPolicy
from .metrics import summarize_period_returns
from .portfolio_weights import normalize_position_weights

PositionExitPolicy = Literal["period", "strict", "ffill", "delay"]


@dataclass(frozen=True)
class PositionBacktestConfig:
    price_col: str = "close"
    transaction_cost_bps: float = 0.0
    trading_days_per_year: int = 252
    long_only: bool = True
    preserve_gross_exposure: bool = False
    exit_price_policy: PositionExitPolicy = "period"
    exit_fallback_policy: Literal["ffill", "none"] = "ffill"
    tradable_col: str | None = None


@dataclass(frozen=True)
class PositionBacktestResult:
    net_returns: pd.DataFrame
    gross_returns: pd.DataFrame
    periods: pd.DataFrame
    summary: dict[str, Any]


def _date_series(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    parsed = pd.to_datetime(text, errors="coerce")
    compact = text.str.fullmatch(r"\d{8}")
    if compact.any():
        parsed.loc[compact] = pd.to_datetime(text.loc[compact], format="%Y%m%d", errors="coerce")
    return parsed.dt.normalize()


def _date_value(value: Any) -> pd.Timestamp:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if len(text) == 8 and text.isdigit():
        parsed = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    else:
        parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return pd.NaT
    return pd.Timestamp(parsed).normalize()


def _date_key(value: Any) -> str:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        text = str(value).strip().replace("-", "").replace(".0", "")
        return text
    return pd.Timestamp(timestamp).strftime("%Y%m%d")


def normalize_position_backtest_positions(positions: pd.DataFrame) -> pd.DataFrame:
    assert_positions_by_rebalance_frame(positions)
    out = positions.copy()
    out["rebalance_key"] = _date_series(out["rebalance_date"]).dt.strftime("%Y%m%d")
    out["symbol"] = out["symbol"].astype(str)
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce").fillna(0.0)
    if "side" in out.columns:
        out = out.loc[out["side"].astype(str).str.lower().eq("long")].copy()
    out = out.loc[out["weight"] > 0].copy()
    return out


def normalize_position_backtest_pricing(
    pricing: pd.DataFrame,
    *,
    price_col: str,
    tradable_col: str | None = None,
) -> pd.DataFrame:
    required = {"trade_date", "symbol", price_col}
    missing = sorted(required - set(pricing.columns))
    if missing:
        raise ValueError("Pricing file is missing required column(s): " + ", ".join(missing))
    columns = ["trade_date", "symbol", price_col]
    if tradable_col and tradable_col in pricing.columns:
        columns.append(tradable_col)
    out = pricing[columns].copy()
    out["trade_date"] = _date_series(out["trade_date"])
    out["symbol"] = out["symbol"].astype(str)
    out[price_col] = pd.to_numeric(out[price_col], errors="coerce")
    return out.dropna(subset=["trade_date", "symbol", price_col]).drop_duplicates(
        subset=["trade_date", "symbol"],
        keep="last",
    )


def normalize_position_backtest_periods(periods: pd.DataFrame) -> pd.DataFrame:
    required = {"rebalance_date", "entry_date", "exit_date"}
    missing = sorted(required - set(periods.columns))
    if missing:
        raise ValueError("Periods file is missing required column(s): " + ", ".join(missing))
    out = periods.copy()
    out["rebalance_key"] = _date_series(out["rebalance_date"]).dt.strftime("%Y%m%d")
    out["entry_date_ts"] = _date_series(out["entry_date"])
    out["exit_date_ts"] = _date_series(out["exit_date"])
    out = out.dropna(subset=["rebalance_key", "entry_date_ts", "exit_date_ts"]).copy()
    out = out.sort_values(["entry_date_ts", "rebalance_key"]).reset_index(drop=True)
    if "entry_idx" not in out.columns:
        out["entry_idx"] = range(out.shape[0])
    if "exit_idx" not in out.columns:
        out["exit_idx"] = pd.to_numeric(out["entry_idx"], errors="coerce").fillna(0).astype(int) + 1
    if "planned_exit_idx" not in out.columns:
        out["planned_exit_idx"] = out["exit_idx"]
    if "planned_exit_date" not in out.columns:
        out["planned_exit_date"] = out["exit_date"]
    if "exit_delay_steps" not in out.columns:
        out["exit_delay_steps"] = (
            pd.to_numeric(out["exit_idx"], errors="coerce")
            - pd.to_numeric(out["planned_exit_idx"], errors="coerce")
        ).fillna(0)
    return out


def _price_table(pricing: pd.DataFrame, *, price_col: str) -> pd.DataFrame:
    return pricing.pivot_table(
        index="trade_date",
        columns="symbol",
        values=price_col,
        aggfunc="last",
    ).sort_index()


def _tradable_table(pricing: pd.DataFrame, *, tradable_col: str | None) -> pd.DataFrame | None:
    if not tradable_col or tradable_col not in pricing.columns:
        return None
    table = pricing.pivot_table(
        index="trade_date",
        columns="symbol",
        values=tradable_col,
        aggfunc="last",
    ).sort_index()
    return table.where(table.notna(), False).astype(bool)


def _clean_position_weights(weights: pd.Series, *, preserve_gross_exposure: bool) -> pd.Series:
    if not preserve_gross_exposure:
        return normalize_position_weights(weights)
    if weights is None or weights.empty:
        return pd.Series(dtype=float)
    cleaned = (
        pd.to_numeric(weights, errors="coerce")
        .replace([float("inf"), float("-inf")], pd.NA)
        .dropna()
    )
    cleaned = cleaned.loc[cleaned > 0]
    if cleaned.empty:
        return pd.Series(dtype=float)
    total = float(cleaned.sum())
    if total <= 0:
        return pd.Series(dtype=float)
    if total > 1.0:
        return cleaned / total
    return cleaned.astype(float)


def _weights_for_rebalance(
    positions: pd.DataFrame,
    rebalance_key: str,
    *,
    preserve_gross_exposure: bool,
) -> pd.Series:
    rows = positions.loc[positions["rebalance_key"] == rebalance_key]
    if rows.empty:
        return pd.Series(dtype=float)
    grouped = rows.groupby("symbol")["weight"].sum()
    return _clean_position_weights(grouped, preserve_gross_exposure=preserve_gross_exposure)


def _valid_period_prices(
    *,
    weights: pd.Series,
    price_table: pd.DataFrame,
    entry_date: pd.Timestamp,
    exit_date: pd.Timestamp,
    preserve_gross_exposure: bool,
) -> tuple[pd.Series, pd.Series, pd.Series, int]:
    if entry_date not in price_table.index or exit_date not in price_table.index:
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float), len(weights)
    entry_prices = price_table.loc[entry_date].reindex(weights.index)
    exit_prices = price_table.loc[exit_date].reindex(weights.index)
    valid = entry_prices.notna() & exit_prices.notna()
    missing = int((~valid).sum())
    clean_weights = _clean_position_weights(
        weights.loc[valid],
        preserve_gross_exposure=preserve_gross_exposure,
    )
    return (
        clean_weights,
        entry_prices.loc[clean_weights.index],
        exit_prices.loc[clean_weights.index],
        missing,
    )


@dataclass(frozen=True)
class _PositionPeriodPrices:
    target: pd.Series
    entry_prices: pd.Series
    exit_prices: pd.Series
    entry_idx: int
    planned_exit_idx: int
    exit_date: pd.Timestamp
    exit_idx: int
    missing_prices: int


def _idx_for_price_table_date(
    price_table: pd.DataFrame,
    date: Any,
    fallback: int,
) -> int:
    date_ts = _date_value(date)
    if pd.isna(date_ts):
        return int(fallback)
    trade_dates = pd.Index(pd.to_datetime(price_table.index).normalize())
    matches = trade_dates.get_indexer([pd.Timestamp(date_ts).normalize()])
    if len(matches) and int(matches[0]) >= 0:
        return int(matches[0])
    return int(fallback)


def _valid_period_policy_prices(
    *,
    weights: pd.Series,
    price_table: pd.DataFrame,
    entry_date: pd.Timestamp,
    exit_date: pd.Timestamp,
    preserve_gross_exposure: bool,
    entry_idx: int,
    planned_exit_idx: int,
    exit_idx: int,
) -> _PositionPeriodPrices:
    target, entry_prices, exit_prices, missing = _valid_period_prices(
        weights=weights,
        price_table=price_table,
        entry_date=entry_date,
        exit_date=exit_date,
        preserve_gross_exposure=preserve_gross_exposure,
    )
    return _PositionPeriodPrices(
        target=target,
        entry_prices=entry_prices,
        exit_prices=exit_prices,
        entry_idx=entry_idx,
        planned_exit_idx=planned_exit_idx,
        exit_date=exit_date,
        exit_idx=exit_idx,
        missing_prices=missing,
    )


def _resolve_exit_policy_prices(
    *,
    weights: pd.Series,
    price_table: pd.DataFrame,
    tradable_table: pd.DataFrame | None,
    entry_date: pd.Timestamp,
    entry_idx: int,
    planned_exit_idx: int,
    config: PositionBacktestConfig,
) -> _PositionPeriodPrices:
    if entry_date not in price_table.index:
        empty = pd.Series(dtype=float)
        return _PositionPeriodPrices(
            empty,
            empty,
            empty,
            entry_idx,
            planned_exit_idx,
            entry_date,
            planned_exit_idx,
            len(weights),
        )
    entry_prices = price_table.loc[entry_date].reindex(weights.index)
    valid_entry = entry_prices.notna()
    target = _clean_position_weights(
        weights.loc[valid_entry],
        preserve_gross_exposure=config.preserve_gross_exposure,
    )
    if target.empty:
        empty = pd.Series(dtype=float)
        return _PositionPeriodPrices(
            empty,
            empty,
            empty,
            entry_idx,
            planned_exit_idx,
            entry_date,
            planned_exit_idx,
            len(weights),
        )

    trade_dates = list(price_table.index)
    date_to_idx = {pd.Timestamp(date): idx for idx, date in enumerate(trade_dates)}
    exit_policy = ExitPolicy(
        config.exit_price_policy,
        config.exit_fallback_policy,
        config.price_col,
    )
    exit_prices, exit_idx = exit_policy.resolve_exit_prices(
        list(target.index),
        planned_exit_idx,
        price_table=price_table,
        tradable_table=tradable_table,
        trade_dates=trade_dates,
        date_to_idx=date_to_idx,
    )
    target = _clean_position_weights(
        target.reindex(exit_prices.index).dropna(),
        preserve_gross_exposure=config.preserve_gross_exposure,
    )
    missing = int(len(weights) - len(target))
    return _PositionPeriodPrices(
        target=target,
        entry_prices=entry_prices.reindex(target.index),
        exit_prices=exit_prices.reindex(target.index),
        entry_idx=entry_idx,
        planned_exit_idx=planned_exit_idx,
        exit_date=pd.Timestamp(trade_dates[int(exit_idx)]),
        exit_idx=int(exit_idx),
        missing_prices=missing,
    )


def _planned_exit_idx_for_price_table(period: Any, price_table: pd.DataFrame) -> int:
    planned_date_raw = getattr(period, "planned_exit_date", None)
    if planned_date_raw is None:
        planned_date_raw = getattr(period, "exit_date_ts", None)
    return _idx_for_price_table_date(
        price_table,
        planned_date_raw,
        int(period.planned_exit_idx),
    )


def _resolve_position_period_prices(
    *,
    weights: pd.Series,
    period: Any,
    price_table: pd.DataFrame,
    tradable_table: pd.DataFrame | None,
    config: PositionBacktestConfig,
) -> _PositionPeriodPrices:
    entry_date = pd.Timestamp(period.entry_date_ts)
    entry_idx = _idx_for_price_table_date(price_table, entry_date, int(period.entry_idx))
    planned_exit_idx = _planned_exit_idx_for_price_table(period, price_table)
    if config.exit_price_policy == "period":
        exit_date = pd.Timestamp(period.exit_date_ts)
        return _valid_period_policy_prices(
            weights=weights,
            price_table=price_table,
            entry_date=entry_date,
            exit_date=exit_date,
            preserve_gross_exposure=config.preserve_gross_exposure,
            entry_idx=entry_idx,
            planned_exit_idx=planned_exit_idx,
            exit_idx=_idx_for_price_table_date(price_table, exit_date, int(period.exit_idx)),
        )
    return _resolve_exit_policy_prices(
        weights=weights,
        price_table=price_table,
        tradable_table=tradable_table,
        entry_date=entry_date,
        entry_idx=entry_idx,
        planned_exit_idx=planned_exit_idx,
        config=config,
    )


def _period_info_records(periods: pd.DataFrame) -> list[dict[str, Any]]:
    records = []
    for row in periods.to_dict("records"):
        record = dict(row)
        for column in ("entry_date", "planned_exit_date", "exit_date", "rebalance_date"):
            if column in record:
                record[column] = pd.to_datetime(record[column], errors="coerce")
        records.append(record)
    return records


@dataclass
class _PositionBacktestState:
    weights: pd.Series | None = None
    entry_prices: pd.Series | None = None
    entry_date: pd.Timestamp | None = None


def _holding_days(
    period: Any,
    *,
    entry_idx: int | None = None,
    exit_idx: int | None = None,
) -> int | None:
    if hasattr(period, "exit_idx") and hasattr(period, "entry_idx"):
        resolved_entry_idx = int(period.entry_idx) if entry_idx is None else int(entry_idx)
        resolved_exit_idx = int(period.exit_idx) if exit_idx is None else int(exit_idx)
        return resolved_exit_idx - resolved_entry_idx
    return None


def _compute_cash_aware_trade_summary(
    prev_weights: pd.Series | None,
    prev_prices: pd.Series | None,
    prev_date: pd.Timestamp | None,
    target_weights: pd.Series,
    entry_date: pd.Timestamp,
    *,
    price_table: pd.DataFrame,
) -> tuple[float, float, float, pd.Series]:
    target_clean = _clean_position_weights(target_weights, preserve_gross_exposure=True)
    if target_clean.empty:
        return 0.0, 0.0, 0.0, pd.Series(dtype=float)

    if prev_weights is None or prev_weights.empty:
        trade_weights = target_clean.copy()
        traded = float(trade_weights.abs().sum())
        return traded, traded, 0.0, trade_weights

    prev_clean = _clean_position_weights(prev_weights, preserve_gross_exposure=True)
    drift_weights = prev_clean
    if prev_prices is not None and prev_date is not None:
        prev_prices_valid = prev_prices.reindex(prev_clean.index)
        prev_prices_valid = prev_prices_valid[prev_prices_valid.notna()]
        if not prev_prices_valid.empty and entry_date in price_table.index:
            prev_clean = prev_clean.reindex(prev_prices_valid.index).dropna()
            current_prices = price_table.loc[entry_date, prev_prices_valid.index]
            valid_prev = current_prices.notna()
            prev_prices_valid = prev_prices_valid[valid_prev]
            current_prices = current_prices[valid_prev]
            prev_clean = prev_clean.reindex(prev_prices_valid.index).dropna()
            if not prev_prices_valid.empty and not prev_clean.empty:
                position_values = prev_clean * (current_prices / prev_prices_valid)
                cash_value = max(0.0, 1.0 - float(prev_clean.sum()))
                total_value = cash_value + float(position_values.sum())
                if total_value > 0:
                    drift_weights = (position_values / total_value).astype(float)

    all_ids = drift_weights.index.union(target_clean.index)
    drift_aligned = drift_weights.reindex(all_ids).fillna(0.0)
    target_aligned = target_clean.reindex(all_ids).fillna(0.0)
    trade_weights = target_aligned - drift_aligned
    entry_turnover = float(trade_weights.clip(lower=0.0).sum())
    exit_turnover = float((-trade_weights.clip(upper=0.0)).sum())
    turnover = 0.5 * float(trade_weights.abs().sum())
    return turnover, entry_turnover, exit_turnover, trade_weights


def _build_period_result_row(
    *,
    period: Any,
    rebalance_key: str,
    entry_date: pd.Timestamp,
    entry_idx: int,
    planned_exit_idx: int,
    exit_date: pd.Timestamp,
    exit_idx: int,
    target: pd.Series,
    entry_prices: pd.Series,
    exit_prices: pd.Series,
    missing_prices: int,
    state: _PositionBacktestState,
    table: pd.DataFrame,
    cost_model: BpsCostModel,
    preserve_gross_exposure: bool,
) -> tuple[dict[str, Any], _PositionBacktestState]:
    gross = float((((exit_prices / entry_prices) - 1.0) * target).sum())
    if preserve_gross_exposure:
        turnover, entry_turnover, exit_turnover, _ = _compute_cash_aware_trade_summary(
            state.weights,
            state.entry_prices,
            state.entry_date,
            target,
            entry_date,
            price_table=table,
        )
    else:
        turnover, entry_turnover, exit_turnover, _ = _compute_trade_summary(
            state.weights,
            state.entry_prices,
            state.entry_date,
            target,
            entry_date,
            price_table=table,
        )
    fee_cost = cost_model.cost(
        turnover,
        is_initial=state.weights is None,
        side="long",
        entry_turnover=entry_turnover,
        exit_turnover=exit_turnover,
        holding_days=_holding_days(period, entry_idx=entry_idx, exit_idx=exit_idx),
        gross_exposure=float(target.abs().sum()),
    )
    gross_exposure = float(target.abs().sum())
    row = period._asdict()
    row.update(
        {
            "rebalance_date": int(rebalance_key),
            "entry_date": entry_date,
            "entry_idx": int(entry_idx),
            "planned_exit_idx": int(planned_exit_idx),
            "exit_date": exit_date,
            "exit_idx": int(exit_idx),
            "exit_delay_steps": int(exit_idx - planned_exit_idx),
            "turnover": float(turnover),
            "fee_cost": float(fee_cost),
            "slippage_cost": 0.0,
            "total_cost": float(fee_cost),
            "gross_return": gross,
            "net_return": gross - fee_cost,
            "gross_exposure": gross_exposure,
            "cash_weight": max(0.0, 1.0 - gross_exposure),
            "position_count": int(target.shape[0]),
            "missing_price_count": int(missing_prices),
        }
    )
    next_state = _PositionBacktestState(target, entry_prices, entry_date)
    return row, next_state


def _evaluate_position_periods(
    *,
    positions: pd.DataFrame,
    periods: pd.DataFrame,
    table: pd.DataFrame,
    tradable_table: pd.DataFrame | None,
    config: PositionBacktestConfig,
    cost_model: BpsCostModel,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    state = _PositionBacktestState()
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for period in periods.itertuples(index=False):
        rebalance_key = str(period.rebalance_key)
        entry_date = pd.Timestamp(period.entry_date_ts)
        target = _weights_for_rebalance(
            positions,
            rebalance_key,
            preserve_gross_exposure=config.preserve_gross_exposure,
        )
        if target.empty:
            skipped.append({"rebalance_date": rebalance_key, "reason": "missing_positions"})
            continue
        prices = _resolve_position_period_prices(
            weights=target,
            price_table=table,
            tradable_table=tradable_table,
            period=period,
            config=config,
        )
        if prices.target.empty:
            skipped.append({"rebalance_date": rebalance_key, "reason": "missing_prices"})
            continue
        row, state = _build_period_result_row(
            period=period,
            rebalance_key=rebalance_key,
            entry_date=entry_date,
            entry_idx=prices.entry_idx,
            planned_exit_idx=prices.planned_exit_idx,
            exit_date=prices.exit_date,
            exit_idx=prices.exit_idx,
            target=prices.target,
            entry_prices=prices.entry_prices,
            exit_prices=prices.exit_prices,
            missing_prices=prices.missing_prices,
            state=state,
            table=table,
            cost_model=cost_model,
            preserve_gross_exposure=config.preserve_gross_exposure,
        )
        rows.append(row)
    return pd.DataFrame(rows), skipped


def _summarize_position_backtest(
    period_frame: pd.DataFrame,
    *,
    skipped: list[dict[str, Any]],
    config: PositionBacktestConfig,
) -> dict[str, Any]:
    returns = pd.Series(
        period_frame["net_return"].to_numpy(dtype=float),
        index=pd.to_datetime(period_frame["exit_date"]),
        name="net_return",
    )
    stats = summarize_period_returns(
        returns,
        _period_info_records(period_frame),
        int(config.trading_days_per_year),
    )
    stats.update(
        {
            "avg_turnover": float(period_frame["turnover"].mean()),
            "avg_cost_drag": float(period_frame["total_cost"].mean()),
            "avg_fee_drag": float(period_frame["fee_cost"].mean()),
            "avg_slippage_drag": float(period_frame["slippage_cost"].mean()),
            "avg_gross_exposure": float(period_frame["gross_exposure"].mean()),
            "avg_cash_weight": float(period_frame["cash_weight"].mean()),
            "mode": "long_only" if config.long_only else "positions",
            "weighting": "positions",
        }
    )
    return {
        "schema": "position_backtest.v1",
        "config": {
            "price_col": config.price_col,
            "transaction_cost_bps": config.transaction_cost_bps,
            "trading_days_per_year": config.trading_days_per_year,
            "long_only": config.long_only,
            "preserve_gross_exposure": config.preserve_gross_exposure,
            "exit_price_policy": config.exit_price_policy,
            "exit_fallback_policy": config.exit_fallback_policy,
            "tradable_col": config.tradable_col,
        },
        "stats": stats,
        "periods": int(period_frame.shape[0]),
        "skipped_periods": skipped,
    }


def _format_position_backtest_outputs(
    period_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    net = pd.DataFrame(
        {
            "period_end": pd.to_datetime(period_frame["exit_date"]).dt.strftime("%Y-%m-%d"),
            "net_return": period_frame["net_return"],
        }
    )
    gross = pd.DataFrame(
        {
            "period_end": pd.to_datetime(period_frame["exit_date"]).dt.strftime("%Y-%m-%d"),
            "gross_return": period_frame["gross_return"],
        }
    )
    periods = period_frame.drop(columns=["entry_date_ts", "exit_date_ts"], errors="ignore")
    for column in ("entry_date", "planned_exit_date", "exit_date"):
        if column in periods.columns:
            periods[column] = _date_series(periods[column]).dt.strftime("%Y%m%d")
    return net, gross, periods


def run_position_backtest(
    *,
    positions: pd.DataFrame,
    pricing: pd.DataFrame,
    periods: pd.DataFrame,
    config: PositionBacktestConfig,
) -> PositionBacktestResult:
    normalized_positions = normalize_position_backtest_positions(positions)
    normalized_pricing = normalize_position_backtest_pricing(
        pricing,
        price_col=config.price_col,
        tradable_col=config.tradable_col,
    )
    normalized_periods = normalize_position_backtest_periods(periods)
    table = _price_table(normalized_pricing, price_col=config.price_col)
    tradable_table = _tradable_table(normalized_pricing, tradable_col=config.tradable_col)
    period_frame, skipped = _evaluate_position_periods(
        positions=normalized_positions,
        periods=normalized_periods,
        table=table,
        tradable_table=tradable_table,
        config=config,
        cost_model=BpsCostModel(float(config.transaction_cost_bps)),
    )
    if period_frame.empty:
        raise ValueError("Position backtest produced no valid periods.")
    summary = _summarize_position_backtest(period_frame, skipped=skipped, config=config)
    net, gross, formatted_periods = _format_position_backtest_outputs(period_frame)
    return PositionBacktestResult(
        net_returns=net,
        gross_returns=gross,
        periods=formatted_periods,
        summary=summary,
    )


def add_position_backtest_args(parser: Any) -> Any:
    from .position_backtest_cli import add_position_backtest_args as _add_args

    return _add_args(parser)


def run(args: Any) -> PositionBacktestResult:
    from .position_backtest_cli import run as _run

    return _run(args)
