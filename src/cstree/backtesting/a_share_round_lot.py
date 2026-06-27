from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

WeightingMode = Literal["equal", "sqrt_liquidity", "capped_sqrt_liquidity"]


@dataclass(frozen=True)
class RoundLotVariant:
    target_holdings: int
    liquidity_floor_q: float = 0.0
    weighting: WeightingMode = "equal"
    industry_cap: int = 3
    max_weight: float = 0.1
    min_notional: float = 0.0

    @property
    def name(self) -> str:
        return (
            f"h{self.target_holdings}_liq{int(self.liquidity_floor_q * 100):02d}_"
            f"{self.weighting}_icap{self.industry_cap}_max{int(self.max_weight * 100):02d}_"
            f"min{int(self.min_notional / 1000)}k"
        )


def _numeric_series(values: Any, *, index: pd.Index | None = None) -> pd.Series:
    series = pd.Series(values, index=index) if index is not None else pd.Series(values)
    numeric = pd.to_numeric(series, errors="coerce")
    if not isinstance(numeric, pd.Series):
        numeric = pd.Series(numeric, index=series.index)
    return numeric.fillna(0.0)


def _series_float(series: pd.Series, key: str, default: float = np.nan) -> float:
    value = series.get(key, default)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def cap_and_redistribute(raw: pd.Series, cap: float) -> pd.Series:
    """Normalize non-negative weights while capping names when the cap is feasible."""
    base = _numeric_series(raw, index=raw.index).clip(lower=0.0)
    if float(base.sum()) <= 0:
        base = pd.Series(1.0, index=base.index)
    weights = base / float(base.sum())
    if cap <= 0 or cap * len(weights) < 1.0 - 1e-12:
        return weights / float(weights.sum())

    fixed = pd.Series(False, index=weights.index)
    for _ in range(50):
        over_cap = (weights > cap + 1e-12) & (~fixed)
        if not bool(over_cap.any()):
            break
        fixed |= over_cap
        weights.loc[fixed] = cap
        residual = 1.0 - float(weights.loc[fixed].sum())
        free = ~fixed
        if residual <= 1e-12 or not bool(free.any()):
            break
        free_base = base.loc[free]
        if float(free_base.sum()) <= 0:
            weights.loc[free] = residual / int(free.sum())
        else:
            weights.loc[free] = residual * free_base / float(free_base.sum())
    return weights / float(weights.sum())


def _cap_and_redistribute_allow_cash(raw: pd.Series, cap: float) -> pd.Series:
    """Cap weights and leave cash when too few names make the cap infeasible."""
    base = _numeric_series(raw, index=raw.index).clip(lower=0.0)
    if base.empty:
        return base
    if float(base.sum()) <= 0:
        base = pd.Series(1.0, index=base.index)
    if cap <= 0:
        weights = base / float(base.sum())
        return weights / float(weights.sum())

    target_total = min(1.0, float(cap) * len(base))
    weights = target_total * base / float(base.sum())
    fixed = pd.Series(False, index=weights.index)
    for _ in range(50):
        over_cap = (weights > cap + 1e-12) & (~fixed)
        if not bool(over_cap.any()):
            break
        fixed |= over_cap
        weights.loc[fixed] = cap
        residual = target_total - float(weights.loc[fixed].sum())
        free = ~fixed
        if residual <= 1e-12 or not bool(free.any()):
            weights.loc[free] = 0.0
            break
        free_base = base.loc[free]
        if float(free_base.sum()) <= 0:
            weights.loc[free] = residual / int(free.sum())
        else:
            weights.loc[free] = residual * free_base / float(free_base.sum())
    return weights.clip(lower=0.0, upper=cap)


def _round_lot_account_diagnostics(
    *,
    target_names: int,
    equity: float,
    cash_buffer: float,
) -> dict[str, Any]:
    return {
        "target_names": target_names,
        "eligible_names": 0,
        "allocated_names": 0,
        "skipped_no_price": 0,
        "skipped_high_price": 0,
        "skipped_one_lot_gt_target": 0,
        "skipped_min_notional": 0,
        "target_notional_sum": 0.0,
        "target_weight_sum": 0.0,
        "actual_notional_sum": 0.0,
        "actual_weight_sum": 0.0,
        "max_actual_weight": 0.0,
        "cash_left": float(max(equity - cash_buffer, 0.0)) if equity > 0 else 0.0,
        "cash_buffer": float(max(cash_buffer, 0.0)),
        "abs_weight_error_sum": 0.0,
        "redistribution_rounds": 0,
    }


def _round_lot_account_candidates(
    *,
    targets: pd.DataFrame,
    entry_prices: pd.Series,
    diagnostics: dict[str, Any],
    equity: float,
    round_lot: int,
    max_weight: float,
    target_weight_col: str,
) -> pd.DataFrame:
    max_notional = float(equity) * float(max_weight) if max_weight > 0 else float(equity)
    candidates: list[dict[str, float | str]] = []
    for _, row in targets.iterrows():
        symbol = str(row["symbol"])
        price = _series_float(entry_prices, symbol)
        if not math.isfinite(price) or price <= 0:
            diagnostics["skipped_no_price"] += 1
            continue
        one_lot = price * round_lot
        if max_weight > 0 and one_lot > max_notional + 1e-9:
            diagnostics["skipped_high_price"] += 1
            diagnostics["skipped_one_lot_gt_target"] += 1
            continue
        candidates.append(
            {
                "symbol": symbol,
                "price": float(price),
                "one_lot": float(one_lot),
                "raw_weight": max(float(row[target_weight_col]), 0.0),
            }
        )
    return pd.DataFrame(candidates)


def _eligible_round_lot_account_frame(
    *,
    frame: pd.DataFrame,
    diagnostics: dict[str, Any],
    equity: float,
    min_notional: float,
    max_weight: float,
) -> pd.DataFrame:
    active = frame.copy()
    final_weights = pd.Series(dtype=float)
    for _ in range(len(frame) + 1):
        if active.empty:
            return active
        weights = _cap_and_redistribute_allow_cash(
            pd.Series(active["raw_weight"].to_numpy(dtype=float), index=active.index),
            max_weight,
        )
        target_notional = float(equity) * weights
        too_small = target_notional < float(min_notional)
        too_expensive = active["one_lot"].astype(float) > target_notional + 1e-9
        drop_mask = too_small | too_expensive
        if not bool(drop_mask.any()):
            final_weights = weights
            break
        diagnostics["skipped_min_notional"] += int((too_small & ~too_expensive).sum())
        diagnostics["skipped_high_price"] += int(too_expensive.sum())
        diagnostics["skipped_one_lot_gt_target"] += int(too_expensive.sum())
        active = active.loc[~drop_mask].copy()
    if active.empty or final_weights.empty:
        return active.iloc[0:0].copy()

    active = active.copy()
    active["target_weight"] = final_weights.reindex(active.index).fillna(0.0).to_numpy()
    active["target_notional"] = float(equity) * active["target_weight"].astype(float)
    diagnostics["eligible_names"] = len(active)
    diagnostics["target_weight_sum"] = float(active["target_weight"].sum())
    diagnostics["target_notional_sum"] = float(active["target_notional"].sum())
    return active


def _floor_round_lot_account_allocations(
    *,
    active: pd.DataFrame,
    investable_cash: float,
    round_lot: int,
) -> tuple[dict[str, int], dict[str, float], float]:
    allocation: dict[str, int] = {}
    actual_notional: dict[str, float] = {}
    cash = float(investable_cash)
    for _, row in active.sort_values("target_weight", ascending=False).iterrows():
        symbol = str(row["symbol"])
        lots = math.floor(float(row["target_notional"]) / float(row["one_lot"]))
        shares = lots * round_lot
        notional = shares * float(row["price"])
        if shares <= 0 or notional <= 0 or notional > cash + 1e-9:
            continue
        allocation[symbol] = int(shares)
        actual_notional[symbol] = float(notional)
        cash -= float(notional)
    return allocation, actual_notional, cash


def _redistribute_round_lot_account_cash(
    *,
    active: pd.DataFrame,
    allocation: dict[str, int],
    actual_notional: dict[str, float],
    diagnostics: dict[str, Any],
    cash: float,
    max_notional: float,
    max_weight: float,
    round_lot: int,
) -> float:
    active_by_symbol = active.set_index("symbol", drop=False)
    while allocation:
        affordable: list[tuple[float, float, str]] = []
        for symbol, row in active_by_symbol.iterrows():
            one_lot = float(row["one_lot"])
            current = actual_notional.get(str(symbol), 0.0)
            if one_lot > cash + 1e-9:
                continue
            if max_weight > 0 and current + one_lot > max_notional + 1e-9:
                continue
            shortage = float(row["target_notional"]) - current
            affordable.append((shortage, float(row["target_weight"]), str(symbol)))
        if not affordable:
            break
        affordable.sort(reverse=True)
        _shortage, _weight, symbol = affordable[0]
        row = active_by_symbol.loc[symbol]
        price = float(row["price"])
        one_lot = float(row["one_lot"])
        allocation[symbol] = allocation.get(symbol, 0) + round_lot
        actual_notional[symbol] = actual_notional.get(symbol, 0.0) + one_lot
        cash -= one_lot
        diagnostics["redistribution_rounds"] += 1
        if cash < min(float(active_by_symbol["one_lot"].min()), price * round_lot) - 1e-9:
            break
    return float(cash)


def _finish_round_lot_account_diagnostics(
    *,
    active: pd.DataFrame,
    allocation: dict[str, int],
    actual_notional: dict[str, float],
    diagnostics: dict[str, Any],
    equity: float,
    cash: float,
) -> None:
    actual_sum = float(sum(actual_notional.values()))
    diagnostics["allocated_names"] = len(allocation)
    diagnostics["actual_notional_sum"] = actual_sum
    diagnostics["actual_weight_sum"] = actual_sum / float(equity)
    diagnostics["max_actual_weight"] = (
        float(max(actual_notional.values()) / float(equity)) if actual_notional else 0.0
    )
    diagnostics["cash_left"] = float(cash)
    target_map = {str(row["symbol"]): float(row["target_weight"]) for _, row in active.iterrows()}
    actual_weights = {
        symbol: notional / float(equity) for symbol, notional in actual_notional.items()
    }
    all_symbols = set(target_map) | set(actual_weights)
    diagnostics["abs_weight_error_sum"] = float(
        sum(
            abs(actual_weights.get(symbol, 0.0) - target_map.get(symbol, 0.0))
            for symbol in all_symbols
        )
    )


def select_round_lot_targets(
    day: pd.DataFrame,
    variant: RoundLotVariant,
    previous_symbols: set[str] | None = None,
) -> pd.DataFrame:
    """Select target names with liquidity, prior-holding, and industry-cap constraints."""
    if day.empty:
        return pd.DataFrame()
    required = {"symbol", "signal_backtest", "medadv20_amount"}
    missing = sorted(required - set(day.columns))
    if missing:
        raise ValueError(f"round-lot target selection missing required columns: {missing}")

    previous_symbols = previous_symbols or set()
    frame = day.copy()
    liquidity = _numeric_series(frame["medadv20_amount"], index=frame.index)
    if variant.liquidity_floor_q > 0 and int(liquidity.notna().sum()) > 1:
        frame = frame[
            (liquidity.isna()) | (liquidity >= liquidity.quantile(variant.liquidity_floor_q))
        ]
        frame = frame.copy()
    frame = cast("pd.DataFrame", cast(Any, frame).sort_values("signal_backtest", ascending=False))
    if "first_industry_name" not in frame.columns:
        frame["first_industry_name"] = "UNKNOWN"

    keep_pool = frame.head(max(variant.target_holdings * 2, variant.target_holdings))
    keep_symbols = set(
        keep_pool.loc[keep_pool["symbol"].isin(list(previous_symbols)), "symbol"].astype(str)
    )
    ordered = pd.concat(
        [
            frame[frame["symbol"].isin(list(keep_symbols))],
            frame[~frame["symbol"].isin(list(keep_symbols))],
        ],
        ignore_index=True,
    )
    ordered = cast("pd.DataFrame", cast(Any, ordered).drop_duplicates("symbol", keep="first"))

    selected: list[pd.Series] = []
    industry_counts: dict[str, int] = {}
    for _, row in ordered.iterrows():
        industry = str(row.get("first_industry_name") or "UNKNOWN")
        if industry_counts.get(industry, 0) >= variant.industry_cap:
            continue
        selected.append(cast("pd.Series", row))
        industry_counts[industry] = industry_counts.get(industry, 0) + 1
        if len(selected) >= variant.target_holdings:
            break
    if not selected:
        return pd.DataFrame()

    out = pd.DataFrame(selected).copy()
    if variant.weighting == "equal":
        raw = pd.Series(1.0, index=out.index)
    else:
        liquidity_raw = _numeric_series(out["medadv20_amount"], index=out.index).clip(lower=1.0)
        raw = pd.Series(np.sqrt(liquidity_raw.to_numpy()), index=out.index)
    out["target_weight"] = cap_and_redistribute(raw, variant.max_weight).to_numpy()
    return out


def allocate_round_lot(
    targets: pd.DataFrame,
    entry_prices: pd.Series,
    *,
    equity: float,
    round_lot: int,
    min_notional: float,
    max_weight: float,
) -> tuple[dict[str, int], dict[str, Any]]:
    """Convert target weights to round-lot holdings and allocation diagnostics."""
    diagnostics: dict[str, Any] = {
        "target_names": len(targets),
        "skipped_no_price": 0,
        "skipped_one_lot_gt_target": 0,
        "skipped_min_notional": 0,
        "target_notional_sum": 0.0,
        "actual_notional_sum": 0.0,
        "abs_weight_error_sum": 0.0,
    }
    if equity <= 0 or round_lot <= 0 or targets.empty:
        return {}, diagnostics

    rows: list[tuple[str, int, float, float]] = []
    cash_budget = float(equity)
    for _, row in targets.iterrows():
        symbol = str(row["symbol"])
        price = _series_float(entry_prices, symbol)
        if not math.isfinite(price) or price <= 0:
            diagnostics["skipped_no_price"] += 1
            continue
        target_weight = min(float(row["target_weight"]), max_weight)
        target_notional = float(equity) * target_weight
        one_lot = price * round_lot
        diagnostics["target_notional_sum"] += target_notional
        if target_notional < min_notional:
            diagnostics["skipped_min_notional"] += 1
            continue
        if one_lot > target_notional:
            diagnostics["skipped_one_lot_gt_target"] += 1
            continue
        lots = math.floor(target_notional / one_lot)
        if lots <= 0:
            diagnostics["skipped_one_lot_gt_target"] += 1
            continue
        shares = lots * round_lot
        rows.append((symbol, shares, shares * price, target_weight))

    allocation: dict[str, int] = {}
    for symbol, shares, notional, _target_weight in sorted(
        rows, key=lambda item: item[3], reverse=True
    ):
        if notional <= cash_budget + 1e-9:
            allocation[symbol] = int(shares)
            cash_budget -= notional

    actual_sum = sum(
        _series_float(entry_prices, symbol) * qty for symbol, qty in allocation.items()
    )
    diagnostics["actual_notional_sum"] = float(actual_sum)
    target_map = {str(row["symbol"]): float(row["target_weight"]) for _, row in targets.iterrows()}
    actual_weights = {
        symbol: _series_float(entry_prices, symbol) * qty / float(equity)
        for symbol, qty in allocation.items()
    }
    all_symbols = set(target_map) | set(actual_weights)
    diagnostics["abs_weight_error_sum"] = float(
        sum(
            abs(actual_weights.get(symbol, 0.0) - target_map.get(symbol, 0.0))
            for symbol in all_symbols
        )
    )
    return allocation, diagnostics


def allocate_round_lot_account(
    targets: pd.DataFrame,
    entry_prices: pd.Series,
    *,
    equity: float,
    round_lot: int = 100,
    min_notional: float = 5_000.0,
    max_weight: float = 0.10,
    cash_buffer: float = 0.0,
    target_weight_col: str = "target_weight",
) -> tuple[dict[str, int], dict[str, Any]]:
    """Allocate a small A-share account with round lots, caps, and cash reuse."""
    diagnostics = _round_lot_account_diagnostics(
        target_names=len(targets),
        equity=equity,
        cash_buffer=cash_buffer,
    )
    if equity <= 0 or round_lot <= 0 or targets.empty:
        return {}, diagnostics
    if target_weight_col not in targets.columns:
        raise ValueError(f"targets missing required column: {target_weight_col}")

    investable_cash = max(float(equity) - max(float(cash_buffer), 0.0), 0.0)
    diagnostics["cash_left"] = investable_cash
    if investable_cash <= 0:
        return {}, diagnostics

    max_notional = float(equity) * float(max_weight) if max_weight > 0 else float(equity)
    frame = _round_lot_account_candidates(
        targets=targets,
        entry_prices=entry_prices,
        diagnostics=diagnostics,
        equity=equity,
        round_lot=round_lot,
        max_weight=max_weight,
        target_weight_col=target_weight_col,
    )
    if frame.empty:
        return {}, diagnostics

    active = _eligible_round_lot_account_frame(
        frame=frame,
        diagnostics=diagnostics,
        equity=equity,
        min_notional=min_notional,
        max_weight=max_weight,
    )
    if active.empty:
        return {}, diagnostics

    allocation, actual_notional, cash = _floor_round_lot_account_allocations(
        active=active,
        investable_cash=investable_cash,
        round_lot=round_lot,
    )
    cash = _redistribute_round_lot_account_cash(
        active=active,
        allocation=allocation,
        actual_notional=actual_notional,
        diagnostics=diagnostics,
        cash=cash,
        max_notional=max_notional,
        max_weight=max_weight,
        round_lot=round_lot,
    )
    _finish_round_lot_account_diagnostics(
        active=active,
        allocation=allocation,
        actual_notional=actual_notional,
        diagnostics=diagnostics,
        equity=equity,
        cash=cash,
    )
    return allocation, diagnostics


def portfolio_value(holdings: dict[str, int], prices: pd.Series, cash: float) -> float:
    value = float(cash)
    for symbol, quantity in holdings.items():
        price = _series_float(prices, symbol)
        if math.isfinite(price) and price > 0:
            value += price * quantity
    return float(value)
