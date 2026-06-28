"""A-share executable OOS Top-K diagnostics.

This is the package-owned form of the historical strategy-pipeline root probe.
The implementation is intentionally behavior-preserving so the boundary move
does not change the simulation rules.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_RUN = Path(
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


def _date8(value: Any) -> str:
    text = str(value)
    if "-" in text:
        return pd.to_datetime(text).strftime("%Y%m%d")
    return text[:8]


def load_positions(run_dir: Path) -> pd.DataFrame:
    pos = pd.read_csv(run_dir / "positions_by_rebalance_oos.csv")
    pos["rebalance_date"] = pos["rebalance_date"].map(_date8)
    pos["entry_date"] = pos["entry_date"].map(_date8)
    pos["symbol"] = pos["symbol"].astype(str)
    pos["rank"] = pd.to_numeric(pos["rank"], errors="coerce")
    pos = pos.dropna(subset=["rank"])
    return pos.sort_values(["rebalance_date", "rank", "symbol"])


def load_prices(symbols: list[str], run_dir: Path) -> pd.DataFrame:
    frames = []
    cache = run_dir.parents[1] / "cache"
    for sym in symbols:
        path = cache / CACHE_FILE_TEMPLATE.format(symbol=sym)
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        keep = [
            c
            for c in [
                "trade_date",
                "symbol",
                "tr_close",
                "amount",
                "medadv20_amount",
                "is_tradable",
                "is_suspended",
                "is_limit_up",
                "is_limit_down",
                "up_limit",
                "down_limit",
            ]
            if c in df.columns
        ]
        df = df[keep].copy()
        df["trade_date"] = df["trade_date"].map(_date8)
        df["symbol"] = df["symbol"].astype(str)
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No daily price cache files found for OOS symbols")
    return (
        pd.concat(frames, ignore_index=True)
        .dropna(subset=["trade_date", "symbol", "tr_close"])
        .drop_duplicates(["trade_date", "symbol"], keep="last")
    )


def portfolio_value(holdings: dict[str, int], prices: pd.Series, cash: float) -> float:
    value = cash
    for sym, qty in holdings.items():
        px = prices.get(sym, np.nan)
        if pd.notna(px) and px > 0:
            value += float(px) * qty
    return float(value)


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


def _rank_map(candidates: pd.DataFrame) -> dict[str, int]:
    return {str(row["symbol"]): int(row["rank"]) for _, row in candidates.iterrows()}


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


def _trade_notional(
    holdings: dict[str, int], target_alloc: dict[str, int], prices: pd.Series
) -> float:
    traded = 0.0
    for sym in sorted(set(holdings) | set(target_alloc)):
        pxv = prices.get(sym, np.nan)
        if pd.isna(pxv) or pxv <= 0:
            continue
        delta = target_alloc.get(sym, 0) - holdings.get(sym, 0)
        traded += abs(delta) * float(pxv)
    return traded


def _row_value(row: pd.Series, name: str, default: Any = np.nan) -> Any:
    return row.get(name, default) if name in row.index else default


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


def _adv_notional(row: pd.Series) -> float:
    value = _row_value(row, "medadv20_amount", np.nan)
    if pd.isna(value) or float(value) <= 0:
        value = _row_value(row, "amount", np.nan)
    if pd.isna(value) or float(value) <= 0:
        return np.nan
    # Tushare daily amount is conventionally reported in thousand CNY.
    return float(value) * 1000.0


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


def _adv_bucket(adv_notional: float) -> str:
    if not math.isfinite(adv_notional) or adv_notional <= 0:
        return "missing"
    if adv_notional < 10_000_000:
        return "lt_10m"
    if adv_notional < 50_000_000:
        return "10m_50m"
    if adv_notional < 200_000_000:
        return "50m_200m"
    return "gte_200m"


def _turnover_action_order(
    holdings: dict[str, int], target_alloc: dict[str, int], prices: pd.Series
) -> list[tuple[int, str, int, float]]:
    actions = []
    for sym in sorted(set(holdings) | set(target_alloc)):
        pxv = prices.get(sym, np.nan)
        if pd.isna(pxv) or pxv <= 0:
            continue
        delta = target_alloc.get(sym, 0) - holdings.get(sym, 0)
        if delta == 0:
            continue
        priority = 0 if delta < 0 and target_alloc.get(sym, 0) == 0 else 1
        priority = 2 if delta > 0 else priority
        actions.append((priority, sym, delta, float(pxv)))
    return sorted(actions)


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


def _holding_values(holdings: dict[str, int], prices: pd.Series) -> list[float]:
    return [
        float(prices.get(s, np.nan)) * q
        for s, q in holdings.items()
        if pd.notna(prices.get(s, np.nan))
    ]


def _market_rows_by_symbol(px: pd.DataFrame, date: str) -> dict[str, pd.Series]:
    rows = px[px["trade_date"] == date]
    return {str(row["symbol"]): row for _, row in rows.iterrows()}


def _blocked_trade_count(trades: pd.DataFrame) -> int:
    if trades.empty or "blocked_or_capped" not in trades.columns:
        return 0
    return int(trades["blocked_or_capped"].fillna(False).sum())


def _avg_impact_bps(trades: pd.DataFrame) -> float:
    if trades.empty or "impact_bps" not in trades.columns:
        return 0.0
    active = trades[pd.to_numeric(trades["notional"], errors="coerce").fillna(0.0) > 0]
    if active.empty:
        return 0.0
    return float(pd.to_numeric(active["impact_bps"], errors="coerce").fillna(0.0).mean())


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


def compute_stats(daily: pd.DataFrame) -> dict[str, Any]:
    r = pd.to_numeric(daily["daily_return"], errors="coerce").fillna(0.0)
    nav = pd.to_numeric(daily["nav"], errors="coerce").ffill()
    years = len(daily) / 252.0
    total = float(nav.iloc[-1] - 1.0)
    ann = float(nav.iloc[-1] ** (1 / years) - 1) if years > 0 and nav.iloc[-1] > 0 else np.nan
    vol = float(r.std(ddof=1) * np.sqrt(252)) if len(r) > 1 else np.nan
    sharpe = (
        float(r.mean() / r.std(ddof=1) * np.sqrt(252))
        if len(r) > 1 and r.std(ddof=1) > 0
        else np.nan
    )
    dd = nav / nav.cummax() - 1.0
    roll63 = r.rolling(63).mean() / r.rolling(63).std(ddof=1) * np.sqrt(252)
    roll126 = r.rolling(126).mean() / r.rolling(126).std(ddof=1) * np.sqrt(252)
    return {
        "daily_rows": len(daily),
        "start": str(daily["trade_date"].iloc[0]),
        "end": str(daily["trade_date"].iloc[-1]),
        "total_return": total,
        "ann_return": ann,
        "ann_vol": vol,
        "sharpe": sharpe,
        "max_drawdown": float(dd.min()),
        "rolling_sharpe_3m_last": float(roll63.dropna().iloc[-1])
        if roll63.notna().any()
        else np.nan,
        "rolling_sharpe_6m_last": float(roll126.dropna().iloc[-1])
        if roll126.notna().any()
        else np.nan,
    }


def _parse_top_ks(text: str) -> list[int]:
    return [int(item) for item in text.split(",") if item]


def _parse_optional_float(value: str) -> float | None:
    if value.lower() in {"", "none", "null", "off"}:
        return None
    return float(value)


def _parse_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "y", "on"}


def _set_trade_fee_args(args: argparse.Namespace) -> None:
    global USE_DETAILED_FEES, BUY_COMMISSION_BPS, SELL_COMMISSION_BPS, STAMP_TAX_SELL_BPS
    global TRANSFER_FEE_BPS, MIN_COMMISSION_CNY, BUY_SLIPPAGE_BPS, SELL_SLIPPAGE_BPS

    USE_DETAILED_FEES = _parse_bool(args.use_detailed_fees)
    BUY_COMMISSION_BPS = float(args.buy_commission_bps)
    SELL_COMMISSION_BPS = float(args.sell_commission_bps)
    STAMP_TAX_SELL_BPS = float(args.stamp_tax_sell_bps)
    TRANSFER_FEE_BPS = float(args.transfer_fee_bps)
    MIN_COMMISSION_CNY = float(args.min_commission_cny)
    BUY_SLIPPAGE_BPS = float(args.buy_slippage_bps)
    SELL_SLIPPAGE_BPS = float(args.sell_slippage_bps)


def _add_trade_fee_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--use-detailed-fees", default="false")
    parser.add_argument("--buy-commission-bps", type=float, default=BUY_COMMISSION_BPS)
    parser.add_argument("--sell-commission-bps", type=float, default=SELL_COMMISSION_BPS)
    parser.add_argument("--stamp-tax-sell-bps", type=float, default=STAMP_TAX_SELL_BPS)
    parser.add_argument("--transfer-fee-bps", type=float, default=TRANSFER_FEE_BPS)
    parser.add_argument("--min-commission-cny", type=float, default=MIN_COMMISSION_CNY)
    parser.add_argument("--buy-slippage-bps", type=float, default=BUY_SLIPPAGE_BPS)
    parser.add_argument("--sell-slippage-bps", type=float, default=SELL_SLIPPAGE_BPS)


def main() -> None:
    global CAPITAL, ROUND_LOT, COST_BPS, TOP_KS, REBALANCE_STRIDE, MAX_TURNOVER_PER_REBALANCE
    global HOLD_BUFFER_RANK, REALISTIC_DAILY_EXECUTION, ADV_PARTICIPATION_LIMIT, IMPACT_BPS_PER_ADV

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN))
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--capital", type=float, default=DEFAULT_CAPITAL)
    parser.add_argument("--round-lot", type=int, default=DEFAULT_ROUND_LOT)
    parser.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS)
    parser.add_argument("--top-ks", default=",".join(str(x) for x in DEFAULT_TOP_KS))
    parser.add_argument("--rebalance-stride", type=int, default=DEFAULT_REBALANCE_STRIDE)
    parser.add_argument("--max-turnover-per-rebalance", default="none")
    parser.add_argument("--hold-buffer-rank", default="none")
    parser.add_argument("--realistic-daily-execution", default="false")
    parser.add_argument("--adv-participation-limit", default="none")
    parser.add_argument("--impact-bps-per-adv", type=float, default=IMPACT_BPS_PER_ADV)
    _add_trade_fee_args(parser)
    args = parser.parse_args()

    CAPITAL = float(args.capital)
    ROUND_LOT = int(args.round_lot)
    COST_BPS = float(args.cost_bps)
    TOP_KS = _parse_top_ks(args.top_ks)
    REBALANCE_STRIDE = max(1, int(args.rebalance_stride))
    MAX_TURNOVER_PER_REBALANCE = _parse_optional_float(args.max_turnover_per_rebalance)
    parsed_buffer = _parse_optional_float(args.hold_buffer_rank)
    HOLD_BUFFER_RANK = int(parsed_buffer) if parsed_buffer is not None else None
    REALISTIC_DAILY_EXECUTION = _parse_bool(args.realistic_daily_execution)
    ADV_PARTICIPATION_LIMIT = _parse_optional_float(args.adv_participation_limit)
    IMPACT_BPS_PER_ADV = float(args.impact_bps_per_adv)
    _set_trade_fee_args(args)

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "live_executable_500k_oos_topk"
    out_dir.mkdir(parents=True, exist_ok=True)
    pos = load_positions(run_dir)
    px = load_prices(sorted(pos["symbol"].unique()), run_dir)
    summaries = []
    all_diag = []
    for top_k in TOP_KS:
        stats, daily, diag, trades = simulate(pos, px, top_k)
        summaries.append(stats)
        all_diag.append(diag)
        daily.to_csv(out_dir / f"daily_top{top_k}.csv", index=False)
        diag.to_csv(out_dir / f"rebalance_diag_top{top_k}.csv", index=False)
        trades.to_csv(out_dir / f"rebalance_trades_top{top_k}.csv", index=False)
        print(
            "done",
            top_k,
            "sharpe",
            round(stats["sharpe"], 3),
            "ret",
            round(stats["total_return"], 3),
            "cash",
            round(stats["avg_cash_weight_daily"], 3),
        )
    summary = pd.DataFrame(summaries).sort_values("top_k")
    summary.to_csv(out_dir / "topk_summary.csv", index=False)
    pd.concat(all_diag, ignore_index=True).to_csv(out_dir / "rebalance_diag_all.csv", index=False)
    meta = {
        "source_run": str(run_dir),
        "positions_source": "positions_by_rebalance_oos.csv",
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
        "affordability_filter": (
            "drop candidate if one 100-share lot exceeds equal target slot at "
            "current equity; backfill from ranks available in source top15"
        ),
        "cash_redistribution": (
            "floor lots, then add one lot at a time to most-underweight "
            "selected names without breaching min(abs 18%, 1/k * 1.35)"
        ),
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("WROTE", out_dir)


if __name__ == "__main__":
    main()
