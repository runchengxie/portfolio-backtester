from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .a_share_round_lot import (
    RoundLotVariant,
    allocate_round_lot,
    portfolio_value,
    select_round_lot_targets,
)


def trading_dates(pricing: pd.DataFrame) -> list[str]:
    return sorted(pricing["trade_date"].dropna().unique().tolist())


def next_trade_date(date: str, dates: list[str], offset: int = 1) -> str | None:
    try:
        idx = dates.index(date)
    except ValueError:
        later = [d for d in dates if d > date]
        if not later:
            return None
        idx = dates.index(later[0])
        offset = 0
    out = idx + offset
    if out >= len(dates):
        return None
    return dates[out]


def rebalance_dates(scored: pd.DataFrame, periods_oos: pd.DataFrame | None) -> list[str]:
    dates = sorted(scored["trade_date"].dropna().unique().tolist())
    if periods_oos is None:
        return dates
    oos_rebals = set(
        pd.to_datetime(periods_oos["rebalance_date"], errors="coerce").dt.strftime("%Y%m%d")
    )
    matching = [date for date in dates if date in oos_rebals]
    return matching if matching else dates


@dataclass(frozen=True)
class RoundLotSimulationCalendar:
    groups: dict[str, pd.DataFrame]
    price_table: pd.DataFrame
    dates: list[str]
    entry_dates: list[str]
    entry_dates_set: set[str]
    entry_to_rebalance: dict[str, str]


@dataclass
class RoundLotSimulationState:
    holdings: dict[str, int]
    cash: float
    previous_symbols: set[str]
    last_nav: float


@dataclass
class RoundLotSimulationRows:
    daily: list[dict[str, Any]]
    trades: list[dict[str, Any]]
    diagnostics: list[dict[str, Any]]


def _entry_dates_for_rebalances(
    rebalance_dates_: list[str],
    dates: list[str],
    price_table: pd.DataFrame,
) -> list[str]:
    entry_dates: list[str] = []
    for rebalance_date in rebalance_dates_:
        entry_date = next_trade_date(rebalance_date, dates, offset=1)
        if entry_date and entry_date in price_table.index:
            entry_dates.append(entry_date)
    return entry_dates


def prepare_round_lot_simulation_calendar(
    scored: pd.DataFrame,
    pricing: pd.DataFrame,
    industry: pd.DataFrame,
    oos_periods: pd.DataFrame | None,
) -> RoundLotSimulationCalendar:
    sig = scored.merge(industry[["symbol", "first_industry_name"]], on="symbol", how="left")
    sig["first_industry_name"] = sig["first_industry_name"].fillna("UNKNOWN")
    groups = dict(tuple(sig.groupby("trade_date", sort=False)))
    price_table = pricing.pivot(
        index="trade_date", columns="symbol", values="tr_close"
    ).sort_index()
    dates = trading_dates(pricing)
    rdates = rebalance_dates(scored, oos_periods)
    entry_dates = _entry_dates_for_rebalances(rdates, dates, price_table)
    return RoundLotSimulationCalendar(
        groups=groups,
        price_table=price_table,
        dates=dates,
        entry_dates=entry_dates,
        entry_dates_set=set(entry_dates),
        entry_to_rebalance=dict(zip(entry_dates, rdates, strict=False)),
    )


def _holding_values(holdings: dict[str, int], prices: pd.Series) -> list[float]:
    return [
        float(prices.get(symbol, np.nan)) * quantity
        for symbol, quantity in holdings.items()
        if pd.notna(prices.get(symbol, np.nan))
    ]


def _trade_to_target(
    *,
    date: str,
    rebalance_date_: str,
    prices: pd.Series,
    state: RoundLotSimulationState,
    target_alloc: dict[str, int],
    cost_bps: float,
    trade_rows: list[dict[str, Any]],
) -> tuple[float, float]:
    symbols = set(state.holdings) | set(target_alloc)
    traded_notional = 0.0
    cash = state.cash
    for symbol in sorted(symbols):
        px = prices.get(symbol, np.nan)
        if pd.isna(px) or px <= 0:
            continue
        old = state.holdings.get(symbol, 0)
        new = target_alloc.get(symbol, 0)
        delta = new - old
        if delta == 0:
            continue
        notional = float(px) * abs(delta)
        traded_notional += notional
        if delta > 0:
            cash -= float(px) * delta
        else:
            cash += float(px) * abs(delta)
        trade_rows.append(
            {
                "date": date,
                "rebalance_date": rebalance_date_,
                "symbol": symbol,
                "delta_shares": int(delta),
                "price": float(px),
                "notional": notional,
            }
        )
    cost = traded_notional * cost_bps / 10000.0
    state.cash = cash - cost
    state.holdings = {symbol: quantity for symbol, quantity in target_alloc.items() if quantity > 0}
    state.previous_symbols = set(state.holdings)
    return traded_notional, cost


def _rebalance_diag_payload(
    *,
    date: str,
    rebalance_date_: str,
    variant: RoundLotVariant,
    state: RoundLotSimulationState,
    prices: pd.Series,
    equity_before: float,
    traded_notional: float,
    cost: float,
) -> dict[str, Any]:
    equity_after = portfolio_value(state.holdings, prices, state.cash)
    holding_values = _holding_values(state.holdings, prices)
    return {
        "date": date,
        "rebalance_date": rebalance_date_,
        "variant": variant.name,
        "equity_before": equity_before,
        "equity_after": equity_after,
        "trade_notional": traded_notional,
        "transaction_cost": cost,
        "actual_holdings": len(state.holdings),
        "cash_weight_after_trade": state.cash / equity_after if equity_after else np.nan,
        "max_actual_weight": max(holding_values) / equity_after
        if holding_values and equity_after
        else 0.0,
        "hhi_actual_weight": sum((value / equity_after) ** 2 for value in holding_values)
        if equity_after
        else np.nan,
    }


def _apply_rebalance(
    *,
    date: str,
    prices: pd.Series,
    calendar: RoundLotSimulationCalendar,
    variant: RoundLotVariant,
    state: RoundLotSimulationState,
    rows: RoundLotSimulationRows,
    round_lot: int,
    cost_bps: float,
) -> None:
    rebalance_date_ = calendar.entry_to_rebalance[date]
    equity_before = portfolio_value(state.holdings, prices, state.cash)
    day = calendar.groups.get(rebalance_date_)
    if day is None or day.empty:
        return
    targets = select_round_lot_targets(day, variant, state.previous_symbols)
    target_alloc, diag = allocate_round_lot(
        targets,
        prices,
        equity=equity_before,
        round_lot=round_lot,
        min_notional=variant.min_notional,
        max_weight=variant.max_weight,
    )
    traded_notional, cost = _trade_to_target(
        date=date,
        rebalance_date_=rebalance_date_,
        prices=prices,
        state=state,
        target_alloc=target_alloc,
        cost_bps=cost_bps,
        trade_rows=rows.trades,
    )
    diag.update(
        _rebalance_diag_payload(
            date=date,
            rebalance_date_=rebalance_date_,
            variant=variant,
            state=state,
            prices=prices,
            equity_before=equity_before,
            traded_notional=traded_notional,
            cost=cost,
        )
    )
    rows.diagnostics.append(diag)


def _append_daily_row(
    *,
    date: str,
    prices: pd.Series,
    state: RoundLotSimulationState,
    rows: RoundLotSimulationRows,
    capital: float,
) -> None:
    nav = portfolio_value(state.holdings, prices, state.cash)
    daily_return = nav / state.last_nav - 1.0 if state.last_nav > 0 else 0.0
    holding_values = _holding_values(state.holdings, prices)
    rows.daily.append(
        {
            "trade_date": date,
            "nav": nav / capital,
            "portfolio_value": nav,
            "daily_return": daily_return,
            "cash": state.cash,
            "cash_weight": state.cash / nav if nav else np.nan,
            "holdings": len(state.holdings),
            "gross_exposure": sum(holding_values) / nav if nav else np.nan,
            "max_weight": max(holding_values) / nav if holding_values and nav else 0.0,
        }
    )
    state.last_nav = nav


def _run_variant_simulation(
    *,
    calendar: RoundLotSimulationCalendar,
    variant: RoundLotVariant,
    capital: float,
    round_lot: int,
    cost_bps: float,
) -> RoundLotSimulationRows:
    state = RoundLotSimulationState(
        holdings={},
        cash=capital,
        previous_symbols=set(),
        last_nav=capital,
    )
    rows = RoundLotSimulationRows(daily=[], trades=[], diagnostics=[])
    first_entry_date = calendar.entry_dates[0] if calendar.entry_dates else "99999999"
    for date in calendar.dates:
        if date < first_entry_date:
            continue
        prices = calendar.price_table.loc[date]
        if date in calendar.entry_dates_set:
            _apply_rebalance(
                date=date,
                prices=prices,
                calendar=calendar,
                variant=variant,
                state=state,
                rows=rows,
                round_lot=round_lot,
                cost_bps=cost_bps,
            )
        _append_daily_row(date=date, prices=prices, state=state, rows=rows, capital=capital)
    return rows


def compute_round_lot_simulation_stats(daily: pd.DataFrame) -> dict[str, float]:
    if daily.empty:
        return {
            "total_return": np.nan,
            "ann_return": np.nan,
            "ann_vol": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
        }
    r = pd.to_numeric(daily["daily_return"], errors="coerce").fillna(0.0)
    nav = pd.to_numeric(daily["nav"], errors="coerce").ffill()
    total = float(nav.iloc[-1] - 1.0)
    years = len(daily) / 252.0
    ann = float(nav.iloc[-1] ** (1 / years) - 1) if years > 0 and nav.iloc[-1] > 0 else np.nan
    vol = float(r.std(ddof=1) * np.sqrt(252)) if len(r) > 1 else np.nan
    sharpe = (
        float(r.mean() / r.std(ddof=1) * np.sqrt(252))
        if len(r) > 1 and r.std(ddof=1) > 0
        else np.nan
    )
    drawdown = nav / nav.cummax() - 1.0
    return {
        "daily_rows": len(daily),
        "total_return": total,
        "ann_return": ann,
        "ann_vol": vol,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
        "avg_cash_weight_daily": float(pd.to_numeric(daily["cash_weight"], errors="coerce").mean()),
        "avg_holdings_daily": float(pd.to_numeric(daily["holdings"], errors="coerce").mean()),
        "avg_max_weight_daily": float(pd.to_numeric(daily["max_weight"], errors="coerce").mean()),
    }


def _summary_from_outputs(
    *,
    variant: RoundLotVariant,
    daily: pd.DataFrame,
    trades: pd.DataFrame,
    diag: pd.DataFrame,
    capital: float,
    round_lot: int,
    cost_bps: float,
) -> dict[str, Any]:
    stats = compute_round_lot_simulation_stats(daily)
    return {
        "variant": variant.name,
        "target_holdings": variant.target_holdings,
        "liquidity_floor_q": variant.liquidity_floor_q,
        "weighting": variant.weighting,
        "industry_cap": variant.industry_cap,
        "max_weight": variant.max_weight,
        "min_notional": variant.min_notional,
        "capital": capital,
        "round_lot": round_lot,
        "cost_bps": cost_bps,
        **stats,
        "avg_actual_holdings": float(diag["actual_holdings"].mean()) if not diag.empty else 0.0,
        "min_actual_holdings": int(diag["actual_holdings"].min()) if not diag.empty else 0,
        "avg_cash_after_trade": float(diag["cash_weight_after_trade"].mean())
        if not diag.empty
        else np.nan,
        "avg_max_weight_after_trade": float(diag["max_actual_weight"].mean())
        if not diag.empty
        else np.nan,
        "avg_hhi_after_trade": float(diag["hhi_actual_weight"].mean())
        if not diag.empty
        else np.nan,
        "avg_abs_weight_error_sum": float(diag["abs_weight_error_sum"].mean())
        if not diag.empty
        else np.nan,
        "avg_skipped_one_lot_gt_target": float(diag["skipped_one_lot_gt_target"].mean())
        if not diag.empty
        else np.nan,
        "avg_skipped_min_notional": float(diag["skipped_min_notional"].mean())
        if not diag.empty
        else np.nan,
        "avg_trade_notional": float(diag["trade_notional"].mean()) if not diag.empty else 0.0,
        "trade_count": len(trades),
        "rebalance_count": len(diag),
    }


def simulate_round_lot_variant(
    scored: pd.DataFrame,
    pricing: pd.DataFrame,
    industry: pd.DataFrame,
    variant: RoundLotVariant,
    *,
    capital: float,
    round_lot: int,
    cost_bps: float,
    oos_periods: pd.DataFrame | None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    calendar = prepare_round_lot_simulation_calendar(scored, pricing, industry, oos_periods)
    rows = _run_variant_simulation(
        calendar=calendar,
        variant=variant,
        capital=capital,
        round_lot=round_lot,
        cost_bps=cost_bps,
    )
    daily = pd.DataFrame(rows.daily)
    trades = pd.DataFrame(rows.trades)
    diag = pd.DataFrame(rows.diagnostics)
    summary = _summary_from_outputs(
        variant=variant,
        daily=daily,
        trades=trades,
        diag=diag,
        capital=capital,
        round_lot=round_lot,
        cost_bps=cost_bps,
    )
    return summary, daily, diag
