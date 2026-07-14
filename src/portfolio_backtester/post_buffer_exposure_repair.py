from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

MOMENTUM_COLUMNS = (
    "momentum",
    "momentum_12m",
    "momentum_6m",
    "mom_12m",
    "mom_6m",
    "ret_252",
    "ret_126",
    "ret_120",
    "ret_60",
    "ret_20",
    "ret_5",
)


@dataclass(frozen=True)
class PostBufferExposureRepairConfig:
    strict_guardrail_min_rank: float = 0.70
    bank_fallback_min_rank: float | None = 0.65
    bank_fallback_min_signal: float = 0.0
    exposure_margin: float = 0.003
    bank_industry_name: str = "银行"
    industry_col: str = "first_industry_name"
    signal_col: str = "signal_z"
    guardrail_col: str = "earnings_burst_rank"
    momentum_col: str = "exposure_momentum_z"
    tradable_col: str = "is_tradable"
    max_abs_industry_active: float = 0.20
    max_abs_momentum_active: float = 1.0


@dataclass(frozen=True)
class PostBufferExposureRepairResult:
    positions: pd.DataFrame
    actions: list[dict[str, Any]]


def add_exposure_momentum_z(
    source: pd.DataFrame,
    *,
    momentum_col: str = "exposure_momentum_z",
) -> pd.DataFrame:
    """Add the same momentum z-score used by exposure analysis when source columns exist."""
    work = source.copy()
    if momentum_col in work.columns:
        work["date_i"] = _compact_date_series(work["trade_date"]).astype(int)
        return work
    columns = [column for column in MOMENTUM_COLUMNS if column in work.columns]
    if not columns:
        work[momentum_col] = np.nan
    else:
        components = [pd.to_numeric(work[column], errors="coerce") for column in columns]
        work["_exposure_momentum_raw"] = pd.concat(components, axis=1).mean(axis=1, skipna=True)
        work[momentum_col] = work.groupby("trade_date")["_exposure_momentum_raw"].transform(_zscore)
        work.drop(columns=["_exposure_momentum_raw"], inplace=True)
    work["date_i"] = _compact_date_series(work["trade_date"]).astype(int)
    return work


def repair_post_buffer_exposure(
    positions: pd.DataFrame,
    source: pd.DataFrame,
    breaches: pd.DataFrame,
    *,
    config: PostBufferExposureRepairConfig | None = None,
) -> PostBufferExposureRepairResult:
    cfg = config or PostBufferExposureRepairConfig()
    repaired = normalize_repair_positions(positions)
    source_work = add_exposure_momentum_z(source, momentum_col=cfg.momentum_col)
    source_work = _prepare_repair_source(source_work, cfg)
    actions: list[dict[str, Any]] = []

    for _, breach in _breach_rows(breaches).iterrows():
        if _is_momentum_breach(breach):
            repaired, action = _repair_momentum(
                repaired,
                source_work,
                breach=breach,
                config=cfg,
            )
        elif _is_bank_breach(breach, cfg):
            repaired, action = _repair_bank(
                repaired,
                source_work,
                breach=breach,
                config=cfg,
            )
        else:
            continue
        actions.append(action)
        repaired = normalize_repair_positions(repaired)

    return PostBufferExposureRepairResult(positions=repaired, actions=actions)


def _prepare_repair_source(
    source: pd.DataFrame,
    config: PostBufferExposureRepairConfig,
) -> pd.DataFrame:
    work = source.copy()
    if config.signal_col not in work.columns:
        fallback_signal_col = next(
            (
                column
                for column in ("signal_backtest", "signal_eval", "pred", "signal_z", "signal")
                if column in work.columns
            ),
            None,
        )
        work[config.signal_col] = (
            pd.to_numeric(work[fallback_signal_col], errors="coerce")
            if fallback_signal_col is not None
            else np.nan
        )
    if config.guardrail_col not in work.columns:
        signal = pd.to_numeric(work[config.signal_col], errors="coerce")
        if "trade_date" in work.columns:
            work[config.guardrail_col] = signal.groupby(work["trade_date"]).rank(pct=True)
        else:
            work[config.guardrail_col] = signal.rank(pct=True)
    if config.industry_col not in work.columns:
        work[config.industry_col] = ""
    if config.tradable_col not in work.columns:
        work[config.tradable_col] = True
    return work


def normalize_repair_positions(positions: pd.DataFrame) -> pd.DataFrame:
    out = positions.copy()
    for column in ("rebalance_date", "entry_date"):
        if column not in out.columns:
            raise ValueError(f"positions must include {column}.")
        out[column] = _compact_date_series(out[column]).astype(int)
    if "weight" not in out.columns:
        raise ValueError("positions must include weight.")
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce").fillna(0.0)
    if "signal" not in out.columns:
        out["signal"] = 0.0
    if "rank" not in out.columns:
        out["rank"] = 0
    if "side" not in out.columns:
        out["side"] = "long"
    out = out.loc[out["weight"] > 1e-10].copy()
    grouped = (
        out.groupby(["rebalance_date", "symbol"], as_index=False)
        .agg(
            entry_date=("entry_date", "first"),
            weight=("weight", "sum"),
            signal=("signal", "max"),
            rank=("rank", "min"),
            side=("side", "first"),
        )
        .copy()
    )
    for _, idx in grouped.groupby("rebalance_date").groups.items():
        total = float(grouped.loc[idx, "weight"].sum())
        if total > 0:
            grouped.loc[idx, "weight"] = grouped.loc[idx, "weight"] / total
        signals = pd.to_numeric(grouped.loc[idx, "signal"], errors="coerce").fillna(-1e9)
        grouped.loc[idx, "rank"] = signals.rank(method="first", ascending=False).astype(int)
    grouped["side"] = "long"
    return grouped.sort_values(["rebalance_date", "rank", "symbol"]).reset_index(drop=True)[
        ["rebalance_date", "entry_date", "symbol", "weight", "signal", "rank", "side"]
    ]


def _compact_date_series(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    compact_mask = text.str.fullmatch(r"\d{8}")
    parsed = pd.to_datetime(text, errors="coerce")
    compact = parsed.dt.strftime("%Y%m%d").mask(compact_mask, text)
    return pd.to_numeric(compact, errors="coerce").astype("Int64")


def _zscore(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    std = numeric.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return pd.Series(np.nan, index=values.index)
    return (numeric - numeric.mean()) / std


def _breach_rows(breaches: pd.DataFrame) -> pd.DataFrame:
    if breaches is None or breaches.empty:
        return pd.DataFrame()
    work = breaches.copy()
    if "status" in work.columns:
        work = work.loc[work["status"].astype(str).str.lower() == "breached"]
    if "rebalance_date" in work.columns:
        work["rebalance_date"] = _compact_date_series(work["rebalance_date"]).astype(int)
    return work


def _is_momentum_breach(row: pd.Series) -> bool:
    return str(row.get("check")) == "style_active" and str(row.get("name")) == "momentum"


def _is_bank_breach(row: pd.Series, config: PostBufferExposureRepairConfig) -> bool:
    return (
        str(row.get("check")) == "industry_active"
        and str(row.get("name")) == config.bank_industry_name
    )


def _positions_for_date(positions: pd.DataFrame, date_i: int) -> pd.DataFrame:
    return positions.loc[positions["rebalance_date"] == date_i].copy()


def _source_for_date(source: pd.DataFrame, date_i: int) -> pd.DataFrame:
    return source.loc[source["date_i"] == date_i].copy()


def _ensure_symbol(
    positions: pd.DataFrame,
    *,
    date_i: int,
    entry_date: int,
    symbol: str,
    signal: float,
) -> pd.DataFrame:
    mask = (positions["rebalance_date"] == date_i) & (positions["symbol"] == symbol)
    if mask.any():
        return positions
    date_positions = _positions_for_date(positions, date_i)
    next_rank = int(date_positions["rank"].max() + 1) if not date_positions.empty else 1
    add = pd.DataFrame(
        [
            {
                "rebalance_date": date_i,
                "entry_date": entry_date,
                "symbol": symbol,
                "weight": 0.0,
                "signal": float(signal) if np.isfinite(signal) else 0.0,
                "rank": next_rank,
                "side": "long",
            }
        ]
    )
    return pd.concat([positions, add], ignore_index=True)


def _transfer_weight(
    positions: pd.DataFrame,
    *,
    date_i: int,
    donor_symbols: list[str],
    receiver: str,
    amount: float,
) -> tuple[pd.DataFrame, float, list[dict[str, Any]]]:
    remaining = float(amount)
    moves: list[dict[str, Any]] = []
    for donor in donor_symbols:
        if remaining <= 1e-10:
            break
        donor_mask = (positions["rebalance_date"] == date_i) & (positions["symbol"] == donor)
        if not donor_mask.any():
            continue
        current = positions.loc[donor_mask, "weight"].astype(float)
        available = float(current.sum()) - 1e-8
        if available <= 0:
            continue
        take = min(available, remaining)
        positions.loc[donor_mask, "weight"] = current - current / current.sum() * take
        receiver_mask = (positions["rebalance_date"] == date_i) & (positions["symbol"] == receiver)
        positions.loc[receiver_mask, "weight"] = positions.loc[receiver_mask, "weight"].astype(
            float
        ) + take / max(int(receiver_mask.sum()), 1)
        remaining -= take
        moves.append({"donor": donor, "receiver": receiver, "weight": float(take)})
    return positions, float(amount - remaining), moves


def _repair_bank(
    positions: pd.DataFrame,
    source: pd.DataFrame,
    *,
    breach: pd.Series,
    config: PostBufferExposureRepairConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    date_i = int(breach["rebalance_date"])
    date_positions = _positions_for_date(positions, date_i)
    day = _source_for_date(source, date_i)
    if date_positions.empty or day.empty:
        return positions, {"date": date_i, "repair": "bank", "status": "missing_date"}

    needed_weight = max(
        -config.max_abs_industry_active - float(breach.get("value", 0.0)) + config.exposure_margin,
        0.0,
    )
    strict_pool = _bank_pool(day, min_rank=config.strict_guardrail_min_rank, config=config)
    pool = strict_pool
    fallback_used = False
    if pool.empty and config.bank_fallback_min_rank is not None:
        pool = _bank_pool(
            day,
            min_rank=config.bank_fallback_min_rank,
            config=config,
            min_signal=config.bank_fallback_min_signal,
        )
        fallback_used = True
    if pool.empty or needed_weight <= 0:
        return positions, {
            "date": date_i,
            "repair": "bank",
            "status": "no_candidate" if pool.empty else "no_need",
            "needed_weight": float(needed_weight),
            "fallback_used": bool(fallback_used),
        }

    held_bank_symbols = [
        str(symbol) for symbol in date_positions["symbol"] if str(symbol) in set(pool["symbol"])
    ]
    entry_date = int(date_positions["entry_date"].iloc[0])
    if held_bank_symbols:
        receiver = held_bank_symbols[0]
        receiver_signal = _source_signal(day, receiver, config)
    else:
        candidate = pool.sort_values(
            [config.signal_col, config.guardrail_col],
            ascending=[False, False],
        ).iloc[0]
        receiver = str(candidate["symbol"])
        receiver_signal = float(candidate[config.signal_col])
        positions = _ensure_symbol(
            positions,
            date_i=date_i,
            entry_date=entry_date,
            symbol=receiver,
            signal=receiver_signal,
        )

    donors = _bank_donors(positions, day, date_i, receiver=receiver, config=config)
    positions, moved, moves = _transfer_weight(
        positions,
        date_i=date_i,
        donor_symbols=donors,
        receiver=receiver,
        amount=needed_weight,
    )
    return positions, {
        "date": date_i,
        "repair": "bank",
        "status": "applied" if moved > 0 else "no_donor",
        "needed_weight": float(needed_weight),
        "moved_weight": float(moved),
        "receiver": receiver,
        "receiver_signal": float(receiver_signal),
        "fallback_used": bool(fallback_used),
        "fallback_min_rank": config.bank_fallback_min_rank if fallback_used else None,
        "moves": moves,
    }


def _bank_pool(
    day: pd.DataFrame,
    *,
    min_rank: float,
    config: PostBufferExposureRepairConfig,
    min_signal: float | None = None,
) -> pd.DataFrame:
    pool = day.loc[
        (day[config.industry_col] == config.bank_industry_name)
        & (pd.to_numeric(day[config.guardrail_col], errors="coerce") >= float(min_rank))
    ].copy()
    if min_signal is not None:
        pool = pool.loc[pd.to_numeric(pool[config.signal_col], errors="coerce") > min_signal]
    return pool


def _source_signal(
    day: pd.DataFrame,
    symbol: str,
    config: PostBufferExposureRepairConfig,
) -> float:
    rows = day.loc[day["symbol"] == symbol]
    if rows.empty:
        return 0.0
    value = pd.to_numeric(pd.Series([rows.iloc[-1].get(config.signal_col)]), errors="coerce").iloc[
        0
    ]
    return float(value) if np.isfinite(value) else 0.0


def _bank_donors(
    positions: pd.DataFrame,
    day: pd.DataFrame,
    date_i: int,
    *,
    receiver: str,
    config: PostBufferExposureRepairConfig,
) -> list[str]:
    date_positions = _positions_for_date(positions, date_i)
    merged = date_positions.merge(
        day[["symbol", config.industry_col, config.signal_col]],
        on="symbol",
        how="left",
    )
    donors = merged.loc[
        (merged["symbol"] != receiver)
        & (merged[config.industry_col] != config.bank_industry_name)
        & (merged["weight"] > 1e-6)
    ].copy()
    donors["_signal"] = pd.to_numeric(donors[config.signal_col], errors="coerce")
    donors["_signal"] = donors["_signal"].fillna(pd.to_numeric(donors["signal"], errors="coerce"))
    donors["_signal"] = donors["_signal"].fillna(-999.0)
    return (
        donors.sort_values(["_signal", "weight"], ascending=[True, False])["symbol"]
        .astype(str)
        .tolist()
    )


def _momentum_donor_weight(positions: pd.DataFrame, *, date_i: int, donor_symbol: str) -> float:
    return float(
        positions.loc[
            (positions["rebalance_date"] == date_i) & (positions["symbol"] == donor_symbol),
            "weight",
        ].sum()
    )


def _apply_momentum_moves(
    positions: pd.DataFrame,
    pool: pd.DataFrame,
    donors: pd.DataFrame,
    *,
    date_i: int,
    entry_date: int,
    needed_delta: float,
    config: PostBufferExposureRepairConfig,
) -> tuple[pd.DataFrame, float, float, list[dict[str, Any]]]:
    moved_total = 0.0
    achieved_delta = 0.0
    moves: list[dict[str, Any]] = []
    donor_records = iter(donors.to_dict("records"))
    donor = next(donor_records, None)
    for _, candidate in pool.head(8).iterrows():
        if achieved_delta >= needed_delta:
            break
        receiver = str(candidate["symbol"])
        receiver_z = float(candidate[config.momentum_col])
        positions = _ensure_symbol(
            positions,
            date_i=date_i,
            entry_date=entry_date,
            symbol=receiver,
            signal=float(candidate[config.signal_col]),
        )
        while donor is not None and achieved_delta < needed_delta:
            donor_symbol = str(donor["symbol"])
            donor_z = float(donor[config.momentum_col])
            spread = receiver_z - donor_z
            if spread <= 0:
                donor = next(donor_records, None)
                continue
            donor_weight = _momentum_donor_weight(
                positions,
                date_i=date_i,
                donor_symbol=donor_symbol,
            )
            max_take = max(donor_weight - 1e-8, 0.0)
            if max_take <= 0:
                donor = next(donor_records, None)
                continue
            take = min(max_take, (needed_delta - achieved_delta) / spread)
            positions, moved, move_rows = _transfer_weight(
                positions,
                date_i=date_i,
                donor_symbols=[donor_symbol],
                receiver=receiver,
                amount=take,
            )
            moved_total += moved
            achieved_delta += moved * spread
            for move in move_rows:
                move.update(
                    {
                        "donor_z": float(donor_z),
                        "receiver_z": float(receiver_z),
                        "exposure_delta": float(moved * spread),
                    }
                )
                moves.append(move)
            if moved < max_take - 1e-9:
                break
            donor = next(donor_records, None)
    return positions, moved_total, achieved_delta, moves


def _repair_momentum(
    positions: pd.DataFrame,
    source: pd.DataFrame,
    *,
    breach: pd.Series,
    config: PostBufferExposureRepairConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    date_i = int(breach["rebalance_date"])
    date_positions = _positions_for_date(positions, date_i)
    day = _source_for_date(source, date_i)
    if date_positions.empty or day.empty:
        return positions, {"date": date_i, "repair": "momentum", "status": "missing_date"}

    needed_delta = max(
        -config.max_abs_momentum_active - float(breach.get("value", 0.0)) + config.exposure_margin,
        0.0,
    )
    if needed_delta <= 0:
        return positions, {
            "date": date_i,
            "repair": "momentum",
            "status": "no_need",
            "needed_exposure_delta": float(needed_delta),
        }

    selected = date_positions.merge(
        day[["symbol", config.industry_col, config.signal_col, config.momentum_col]],
        on="symbol",
        how="left",
    )
    donors = selected.loc[
        selected[config.momentum_col].notna() & (selected["weight"] > 1e-6)
    ].sort_values(config.momentum_col)
    pool = _momentum_pool(day, set(date_positions["symbol"]), config=config)
    if donors.empty or pool.empty:
        return positions, {
            "date": date_i,
            "repair": "momentum",
            "status": "no_candidate_or_donor",
            "needed_exposure_delta": float(needed_delta),
        }

    entry_date = int(date_positions["entry_date"].iloc[0])
    positions, moved_total, achieved_delta, moves = _apply_momentum_moves(
        positions,
        pool,
        donors,
        date_i=date_i,
        entry_date=entry_date,
        needed_delta=needed_delta,
        config=config,
    )
    return positions, {
        "date": date_i,
        "repair": "momentum",
        "status": "applied" if moved_total > 0 else "no_move",
        "needed_exposure_delta": float(needed_delta),
        "achieved_exposure_delta_approx": float(achieved_delta),
        "moved_weight": float(moved_total),
        "moves": moves,
    }


def _momentum_pool(
    day: pd.DataFrame,
    held_symbols: set[str],
    *,
    config: PostBufferExposureRepairConfig,
) -> pd.DataFrame:
    tradable = pd.Series(True, index=day.index)
    if config.tradable_col in day.columns:
        tradable = day[config.tradable_col].fillna(True).astype(bool)
    pool = day.loc[
        (
            pd.to_numeric(day[config.guardrail_col], errors="coerce")
            >= config.strict_guardrail_min_rank
        )
        & (~day["symbol"].isin(held_symbols))
        & (day[config.momentum_col].notna())
        & tradable
    ].copy()
    return pool.sort_values([config.momentum_col, config.signal_col], ascending=[False, False])
