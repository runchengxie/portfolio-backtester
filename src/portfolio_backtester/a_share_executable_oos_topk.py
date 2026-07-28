"""A-share executable OOS Top-K diagnostics.

This is the package-owned form of the historical strategy-pipeline root probe.
The implementation is intentionally behavior-preserving so the boundary move
does not change the simulation rules.

The module-level configuration globals (``CAPITAL``, ``ROUND_LOT``,
``USE_DETAILED_FEES``, ...) are intentionally kept on this shell module because
they are monkeypatched by tests and read directly by the engine functions below.
Global-free helpers live in :mod:`portfolio_backtester._aexe_io` and the CLI in
:mod:`portfolio_backtester._aexe_cli`; both are re-exported here so external
imports are unchanged.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from ._aexe_cli import main as main
from ._aexe_io import (
    _adv_bucket as _adv_bucket,
    _adv_notional as _adv_notional,
    _avg_impact_bps as _avg_impact_bps,
    _blocked_trade_count as _blocked_trade_count,
    _date8 as _date8,
    _holding_values as _holding_values,
    _market_rows_by_symbol as _market_rows_by_symbol,
    _rank_map as _rank_map,
    _row_value as _row_value,
    _trade_notional as _trade_notional,
    _turnover_action_order as _turnover_action_order,
    compute_stats as compute_stats,
    load_positions as load_positions,
    load_prices as load_prices,
    portfolio_value as portfolio_value,
)

DEFAULT_RUN = (
    "artifacts/runs/a_share_s_live15_biweekly_max08_min15k_full_oos_20260608_184943_c17c5bce"
)
CACHE_FILE_TEMPLATE = (
    "a_share_tushare_a_share_pit_top800_2015_weekly_"
    "three_statement_core_probe_daily_{symbol}.parquet"
)
DEFAULT_CAPITAL = 500_000.0
DEFAULT_ROUND_LOT = 100
DEFAULT_COST_BPS = 10.0
DEFAULT_TOP_KS = [8, 10, 12, 15]
DEFAULT_REBALANCE_STRIDE = 1
CAPITAL = DEFAULT_CAPITAL
ROUND_LOT = DEFAULT_ROUND_LOT
COST_BPS = DEFAULT_COST_BPS
TOP_KS = DEFAULT_TOP_KS
REBALANCE_STRIDE = DEFAULT_REBALANCE_STRIDE
MAX_TURNOVER_PER_REBALANCE: float | None = None
HOLD_BUFFER_RANK: int | None = None
REALISTIC_DAILY_EXECUTION = False
ADV_PARTICIPATION_LIMIT: float | None = None
IMPACT_BPS_PER_ADV = 50.0
USE_DETAILED_FEES = False
BUY_COMMISSION_BPS = 2.5
SELL_COMMISSION_BPS = 2.5
STAMP_TAX_SELL_BPS = 5.0
TRANSFER_FEE_BPS = 0.1
MIN_COMMISSION_CNY = 5.0
BUY_SLIPPAGE_BPS = 10.0
SELL_SLIPPAGE_BPS = 10.0
# These are intentionally permissive; target feasibility is enforced first.
MAX_WEIGHT_BUFFER = 1.35
ABS_MAX_WEIGHT = 0.18


def _candidate_row(
    row: pd.Series, entry_prices: pd.Series, target_notional: float, target_weight: float
) -> dict[str, Any] | None:
    sym = str(row["symbol"])
    px = float(entry_prices.get(sym, np.nan))
    if not math.isfinite(px) or px <= 0:
        return None
    one_lot = px * ROUND_LOT
    # High-price affordability filter: a name must fit at least one lot inside its target slot.
    if one_lot > target_notional:
        return None
    return {"symbol": sym, "rank": int(row["rank"]), "price": px, "target_weight": target_weight}


def _buffer_keep_symbols(candidates: pd.DataFrame, holdings: dict[str, int]) -> set[str]:
    if HOLD_BUFFER_RANK is None or not holdings:
        return set()
    ranks = _rank_map(candidates)
    return {sym for sym in holdings if ranks.get(sym, HOLD_BUFFER_RANK + 1) <= HOLD_BUFFER_RANK}


def _append_candidate_rows(
    rows: list[dict[str, Any]],
    candidates: pd.DataFrame,
    entry_prices: pd.Series,
    target_notional: float,
    target_weight: float,
    top_k: int,
) -> None:
    selected = {row["symbol"] for row in rows}
    for _, row in candidates.sort_values("rank").iterrows():
        if str(row["symbol"]) in selected:
            continue
        candidate = _candidate_row(row, entry_prices, target_notional, target_weight)
        if candidate is None:
            continue
        rows.append(candidate)
        selected.add(candidate["symbol"])
        if len(rows) >= top_k:
            break


def select_targets(
    candidates: pd.DataFrame,
    entry_prices: pd.Series,
    equity: float,
    top_k: int,
    holdings: dict[str, int] | None = None,
) -> pd.DataFrame:
    selected = []
    target_weight = 1.0 / top_k
    target_notional = equity * target_weight
    hold_symbols = _buffer_keep_symbols(candidates, holdings or {})
    for _, row in (
        candidates[candidates["symbol"].isin(hold_symbols)].sort_values("rank").iterrows()
    ):
        candidate = _candidate_row(row, entry_prices, target_notional, target_weight)
        if candidate is not None:
            selected.append(candidate)
    _append_candidate_rows(
        selected, candidates, entry_prices, target_notional, target_weight, top_k
    )
    return pd.DataFrame(selected)


def allocate_with_redistribution(
    targets: pd.DataFrame,
    prices: pd.Series,
    equity: float,
    top_k: int,
    round_lot: int,
) -> tuple[dict[str, int], dict[str, Any]]:
    alloc: dict[str, int] = {}
    target_weight = 1.0 / top_k
    max_weight = min(ABS_MAX_WEIGHT, target_weight * MAX_WEIGHT_BUFFER)
    skipped = 0
    for _, row in targets.iterrows():
        sym = str(row["symbol"])
        px = float(prices.get(sym, np.nan))
        target_notional = equity * target_weight
        one_lot = px * round_lot
        lots = math.floor(target_notional / one_lot)
        if lots <= 0:
            skipped += 1
            continue
        alloc[sym] = lots * round_lot
    invested = sum(float(prices.get(s, np.nan)) * q for s, q in alloc.items())
    cash = equity - invested
    # Cash redistribution: add one lot at a time to most-underweight names without breaching cap.
    while True:
        choices = []
        for sym, qty in alloc.items():
            px = float(prices.get(sym, np.nan))
            lot_cost = px * round_lot
            current_w = px * qty / equity
            next_w = px * (qty + round_lot) / equity
            if lot_cost <= cash + 1e-9 and next_w <= max_weight + 1e-12:
                choices.append((target_weight - current_w, -lot_cost, sym, lot_cost))
        if not choices:
            break
        _, _, sym, lot_cost = max(choices)
        alloc[sym] += round_lot
        cash -= lot_cost
    invested = sum(float(prices.get(s, np.nan)) * q for s, q in alloc.items())
    actual_w = {s: float(prices.get(s, np.nan)) * q / equity for s, q in alloc.items()}
    diag = {
        "target_names": int(top_k),
        "selected_names": len(targets),
        "actual_holdings": len(alloc),
        "skipped_after_selection": int(skipped),
        "invested_value": float(invested),
        "cash_after_rounding": float(equity - invested),
        "cash_weight_after_rounding": float((equity - invested) / equity),
        "max_actual_weight": float(max(actual_w.values()) if actual_w else 0.0),
        "min_actual_weight": float(min(actual_w.values()) if actual_w else 0.0),
        "abs_weight_error_sum": float(
            sum(abs(actual_w.get(s, 0.0) - target_weight) for s in set(targets["symbol"]))
        ),
        "max_one_lot_cost": float(
            max((float(prices.get(s, np.nan)) * round_lot for s in targets["symbol"]), default=0.0)
        ),
        "median_one_lot_cost": float(
            np.median([float(prices.get(s, np.nan)) * round_lot for s in targets["symbol"]])
            if len(targets)
            else 0.0
        ),
    }
    return alloc, diag


def _is_blocked_trade(row: pd.Series, delta: int) -> bool:
    if not REALISTIC_DAILY_EXECUTION:
        return False
    is_suspended = bool(_row_value(row, "is_suspended", False))
    is_tradable = bool(_row_value(row, "is_tradable", True))
    if is_suspended or not is_tradable:
        return True
    if delta > 0 and bool(_row_value(row, "is_limit_up", False)):
        return True
    return delta < 0 and bool(_row_value(row, "is_limit_down", False))


def _cap_delta_by_participation(delta: int, px: float, row: pd.Series) -> int:
    if ADV_PARTICIPATION_LIMIT is None or not REALISTIC_DAILY_EXECUTION:
        return delta
    adv = _adv_notional(row)
    if not math.isfinite(adv) or adv <= 0:
        return 0
    max_notional = adv * ADV_PARTICIPATION_LIMIT
    max_lots = math.floor(max_notional / (px * ROUND_LOT))
    max_shares = max_lots * ROUND_LOT
    if max_shares <= 0:
        return 0
    return int(math.copysign(min(abs(delta), max_shares), delta))


def _apply_turnover_cap(
    holdings: dict[str, int],
    target_alloc: dict[str, int],
    prices: pd.Series,
    equity: float,
) -> tuple[dict[str, int], dict[str, float]]:
    uncapped = _trade_notional(holdings, target_alloc, prices)
    if MAX_TURNOVER_PER_REBALANCE is None:
        return target_alloc, {"target_trade_notional_uncapped": uncapped, "turnover_budget": np.nan}
    budget = max(0.0, equity * MAX_TURNOVER_PER_REBALANCE)
    capped = holdings.copy()
    used = 0.0
    for _, sym, delta, px in _turnover_action_order(holdings, target_alloc, prices):
        step = ROUND_LOT if delta > 0 else -ROUND_LOT
        remaining = abs(delta)
        while remaining > 0:
            shares = min(ROUND_LOT, remaining)
            cost = shares * px
            if used + cost > budget + 1e-9:
                break
            capped[sym] = capped.get(sym, 0) + (shares if step > 0 else -shares)
            used += cost
            remaining -= shares
        capped = {s: q for s, q in capped.items() if q > 0}
        if used >= budget - 1e-9:
            break
    return capped, {"target_trade_notional_uncapped": uncapped, "turnover_budget": budget}


def _impact_bps(delta: int, px: float, row: pd.Series) -> float:
    if not REALISTIC_DAILY_EXECUTION:
        return 0.0
    adv = _adv_notional(row)
    if not math.isfinite(adv) or adv <= 0:
        return 0.0
    return IMPACT_BPS_PER_ADV * abs(delta) * px / adv


def _trade_cost(notional: float, delta: int, impact_bps: float) -> tuple[float, float]:
    if not USE_DETAILED_FEES:
        total_bps = COST_BPS + impact_bps
        return notional * total_bps / 10000.0, total_bps
    commission_bps = BUY_COMMISSION_BPS if delta > 0 else SELL_COMMISSION_BPS
    commission = max(notional * commission_bps / 10000.0, MIN_COMMISSION_CNY)
    side_bps = BUY_SLIPPAGE_BPS if delta > 0 else SELL_SLIPPAGE_BPS
    stamp_bps = STAMP_TAX_SELL_BPS if delta < 0 else 0.0
    bps_cost = notional * (side_bps + stamp_bps + TRANSFER_FEE_BPS + impact_bps) / 10000.0
    total_cost = commission + bps_cost
    total_bps = total_cost / notional * 10000.0 if notional else 0.0
    return total_cost, total_bps


def _trade_to_target(
    holdings: dict[str, int],
    target_alloc: dict[str, int],
    prices: pd.Series,
    cash: float,
    market_rows: dict[str, pd.Series],
) -> tuple[dict[str, int], float, float, list[dict[str, Any]]]:
    traded = 0.0
    total_cost = 0.0
    trade_rows = []
    next_holdings = holdings.copy()
    for sym in sorted(set(holdings) | set(target_alloc)):
        pxv = prices.get(sym, np.nan)
        if pd.isna(pxv) or pxv <= 0:
            continue
        desired_delta = target_alloc.get(sym, 0) - holdings.get(sym, 0)
        if desired_delta == 0:
            continue
        px = float(pxv)
        row = market_rows.get(sym, pd.Series(dtype="object"))
        delta = desired_delta
        if _is_blocked_trade(row, delta):
            delta = 0
        delta = _cap_delta_by_participation(delta, px, row)
        if delta == 0:
            trade_rows.append(
                {
                    "symbol": sym,
                    "delta_shares": 0,
                    "desired_delta_shares": int(desired_delta),
                    "price": px,
                    "notional": 0.0,
                    "blocked_or_capped": True,
                    "impact_bps": 0.0,
                }
            )
            continue
        notional = abs(delta) * px
        impact_bps = _impact_bps(delta, px, row)
        cost, effective_bps = _trade_cost(notional, delta, impact_bps)
        traded += notional
        total_cost += cost
        cash -= delta * px + cost
        next_holdings[sym] = next_holdings.get(sym, 0) + delta
        trade_rows.append(
            {
                "symbol": sym,
                "delta_shares": int(delta),
                "desired_delta_shares": int(desired_delta),
                "price": px,
                "notional": notional,
                "blocked_or_capped": delta != desired_delta,
                "impact_bps": impact_bps,
                "effective_cost_bps": effective_bps,
            }
        )
    next_holdings = {s: q for s, q in next_holdings.items() if q > 0}
    return next_holdings, cash, total_cost, trade_rows


def _daily_row(
    date: str,
    top_k: int,
    nav_value: float,
    ret: float,
    cash: float,
    hvals: list[float],
) -> dict[str, Any]:
    return {
        "trade_date": date,
        "top_k": top_k,
        "nav": nav_value / CAPITAL,
        "portfolio_value": nav_value,
        "daily_return": ret,
        "cash": cash,
        "cash_weight": cash / nav_value if nav_value else np.nan,
        "holdings": len(hvals),
        "gross_exposure": sum(hvals) / nav_value if nav_value else np.nan,
        "max_weight": max(hvals) / nav_value if hvals and nav_value else 0.0,
    }


def _summary_stats(
    stats: dict[str, Any], daily: pd.DataFrame, diag: pd.DataFrame, trades: pd.DataFrame, top_k: int
) -> dict[str, Any]:
    target_turnover = diag["target_trade_notional_uncapped"] / diag["equity_before"]
    actual_turnover = diag["trade_notional"] / diag["equity_before"]
    stats.update(
        {
            "top_k": top_k,
            "capital": CAPITAL,
            "round_lot": ROUND_LOT,
            "cost_bps": COST_BPS,
            "rebalance_stride": REBALANCE_STRIDE,
            "max_turnover_per_rebalance": MAX_TURNOVER_PER_REBALANCE,
            "hold_buffer_rank": HOLD_BUFFER_RANK,
            "realistic_daily_execution": REALISTIC_DAILY_EXECUTION,
            "adv_participation_limit": ADV_PARTICIPATION_LIMIT,
            "impact_bps_per_adv": IMPACT_BPS_PER_ADV,
            "use_detailed_fees": USE_DETAILED_FEES,
            "buy_commission_bps": BUY_COMMISSION_BPS,
            "sell_commission_bps": SELL_COMMISSION_BPS,
            "stamp_tax_sell_bps": STAMP_TAX_SELL_BPS,
            "transfer_fee_bps": TRANSFER_FEE_BPS,
            "min_commission_cny": MIN_COMMISSION_CNY,
            "buy_slippage_bps": BUY_SLIPPAGE_BPS,
            "sell_slippage_bps": SELL_SLIPPAGE_BPS,
            "rebalance_count": len(diag),
            "trade_count": len(trades),
            "blocked_or_capped_trade_count": _blocked_trade_count(trades),
            "avg_impact_bps": _avg_impact_bps(trades),
            "avg_actual_holdings": float(diag["actual_holdings"].mean()),
            "min_actual_holdings": int(diag["actual_holdings"].min()),
            "avg_selected_names": float(diag["selected_names"].mean()),
            "avg_cash_after_rounding": float(diag["cash_weight_after_rounding"].mean()),
            "avg_cash_after_trade": float(diag["cash_weight_after_trade"].mean()),
            "avg_cash_weight_daily": float(daily["cash_weight"].mean()),
            "avg_max_weight_daily": float(daily["max_weight"].mean()),
            "avg_abs_weight_error_sum": float(diag["abs_weight_error_sum"].mean()),
            "avg_trade_notional": float(diag["trade_notional"].mean()),
            "avg_turnover_on_rebalance": float(actual_turnover.mean()),
            "avg_uncapped_turnover_on_rebalance": float(target_turnover.mean()),
            "turnover_cap_binding_rate": float((actual_turnover < target_turnover - 1e-9).mean()),
            "max_one_lot_cost_seen": float(diag["max_one_lot_cost"].max()),
            "median_one_lot_cost_avg": float(diag["median_one_lot_cost"].mean()),
        }
    )
    return stats


def _active_entry_dates(entry_dates: list[str]) -> list[str]:
    if REBALANCE_STRIDE <= 1:
        return entry_dates
    return entry_dates[::REBALANCE_STRIDE]


def simulate(
    pos: pd.DataFrame, px: pd.DataFrame, top_k: int
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    price_table = px.pivot(index="trade_date", columns="symbol", values="tr_close").sort_index()
    dates = list(price_table.index)
    entry_dates = sorted(pos["entry_date"].unique())
    entry_dates = [d for d in entry_dates if d in price_table.index]
    active_entries = _active_entry_dates(entry_dates)
    entry_set = set(active_entries)
    holdings: dict[str, int] = {}
    cash = CAPITAL
    last_nav = CAPITAL
    daily_rows = []
    diag_rows = []
    trade_rows = []
    previous_rank_by_symbol: dict[str, int] = {}
    first_entry = active_entries[0]
    for date in dates:
        if date < first_entry:
            continue
        prices = price_table.loc[date]
        if date in entry_set:
            equity_before = portfolio_value(holdings, prices, cash)
            candidates = pos[pos["entry_date"] == date].copy()
            targets = select_targets(candidates, prices, equity_before, top_k, holdings)
            target_alloc, diag = allocate_with_redistribution(
                targets, prices, equity_before, top_k, ROUND_LOT
            )
            target_alloc, cap_diag = _apply_turnover_cap(
                holdings, target_alloc, prices, equity_before
            )
            pre_holdings = holdings.copy()
            rank_by_symbol = _rank_map(candidates)
            market_rows = _market_rows_by_symbol(px, date)
            holdings, cash, cost, date_trades = _trade_to_target(
                holdings, target_alloc, prices, cash, market_rows
            )
            for row in date_trades:
                row["date"] = date
                sym = row["symbol"]
                row["side"] = "buy" if row["desired_delta_shares"] > 0 else "sell"
                row["old_shares"] = int(pre_holdings.get(sym, 0))
                row["target_shares"] = int(target_alloc.get(sym, 0))
                row["new_shares"] = int(holdings.get(sym, 0))
                row["old_rank"] = previous_rank_by_symbol.get(sym, np.nan)
                row["new_rank"] = rank_by_symbol.get(sym, np.nan)
                row["is_new_buy"] = row["old_shares"] == 0 and row["desired_delta_shares"] > 0
                row["turnover_contribution"] = (
                    row["notional"] / equity_before if equity_before else np.nan
                )
                mrow = market_rows.get(sym, pd.Series(dtype="object"))
                row["is_suspended"] = bool(_row_value(mrow, "is_suspended", False))
                row["is_limit_up"] = bool(_row_value(mrow, "is_limit_up", False))
                row["is_limit_down"] = bool(_row_value(mrow, "is_limit_down", False))
                row["amount"] = _row_value(mrow, "amount", np.nan)
                row["adv_notional"] = _adv_notional(mrow)
                row["adv_bucket"] = _adv_bucket(row["adv_notional"])
            trade_rows.extend(date_trades)
            traded = sum(row["notional"] for row in date_trades)
            equity_after = portfolio_value(holdings, prices, cash)
            diag.update(cap_diag)
            diag.update(
                {
                    "date": date,
                    "top_k": top_k,
                    "equity_before": equity_before,
                    "equity_after": equity_after,
                    "trade_notional": traded,
                    "transaction_cost": cost,
                    "cash_weight_after_trade": cash / equity_after if equity_after else np.nan,
                }
            )
            diag_rows.append(diag)
            previous_rank_by_symbol = rank_by_symbol
        nav_value = portfolio_value(holdings, prices, cash)
        ret = nav_value / last_nav - 1.0 if last_nav > 0 else 0.0
        hvals = _holding_values(holdings, prices)
        daily_rows.append(_daily_row(date, top_k, nav_value, ret, cash, hvals))
        last_nav = nav_value
    daily = pd.DataFrame(daily_rows)
    diag = pd.DataFrame(diag_rows)
    trades = pd.DataFrame(trade_rows)
    stats = _summary_stats(compute_stats(daily), daily, diag, trades, top_k)
    return stats, daily, diag, trades


if __name__ == "__main__":
    main()
