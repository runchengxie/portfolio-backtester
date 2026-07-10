"""StyleReplica-A80B20-v0 portfolio builder.

Constructs daily A80/B20 portfolios from style_replica signals with:
- A-leg: theme-quota constrained selection (80 slots, 80% capital)
- B-leg: industry-capped low-vol convergence selection (20 slots, 20% capital)
- Position buffer zones (reduces daily turnover)
- Overlap aggregation (A ∩ B → 2% weight)
- Equal-weighted slots

Output conforms to cstree.positions_by_rebalance contract:
    rebalance_date, entry_date, symbol, weight, signal, rank, side

Plus style_replica-specific columns:
    leg, theme, score_a, score_b
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# ── Buffer multipliers ─────────────────────────────────────────────────────────
# Existing holdings are retained until their rank falls below:
#   target_slots * buffer_exit_multiplier
A_BUFFER_EXIT_MULTIPLIER = 1.3
B_BUFFER_EXIT_MULTIPLIER = 1.75  # 35/20
B_BUFFER_ENTRY_RANK = 20
B_MAX_DAILY_REPLACEMENTS = 5


@dataclass
class StyleReplicaPortfolioConfig:
    """Configuration for StyleReplica-A80B20 portfolio construction."""

    # Leg sizes
    a_slots: int = 80
    a_capital_weight: float = 0.80
    b_slots: int = 20
    b_capital_weight: float = 0.20

    # Theme quotas (A-leg)
    theme_quotas: dict[str, int] = field(default_factory=dict)

    # Industry cap (B-leg): max stocks per industry
    b_industry_cap: int = 3

    # Buffer
    a_buffer_exit_multiplier: float = A_BUFFER_EXIT_MULTIPLIER
    b_buffer_exit_rank: int = 35
    b_buffer_entry_rank: int = B_BUFFER_ENTRY_RANK
    b_max_daily_replacements: int = B_MAX_DAILY_REPLACEMENTS

    # Overlap
    overlap_policy: str = "aggregate"  # "aggregate" or "deduplicate"
    normal_slot_weight: float = 0.01
    max_name_weight: float = 0.02

    # Max daily replacements (A+B combined)
    max_daily_replacements: int = 15

    # Model version
    model_version: str = "StyleReplica-A80B20-v0"


def _prepare_signals_frame(signals: pd.DataFrame) -> pd.DataFrame:
    """Normalize signals into the format needed for portfolio construction."""
    df = signals.copy()
    # Ensure date column is datetime
    if "signal_date" in df.columns:
        df["trade_date"] = pd.to_datetime(df["signal_date"], errors="coerce")
        df["signal_date_str"] = df["signal_date"].astype(str)
    elif "trade_date" in df.columns:
        df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
        df["signal_date_str"] = df["trade_date"].dt.strftime("%Y%m%d")

    # Ensure required columns exist with defaults
    for col, default in [
        ("score_a", np.nan),
        ("score_b", np.nan),
        ("leg", None),
        ("theme", None),
        ("industry", None),
    ]:
        if col not in df.columns:
            df[col] = default

    return df.dropna(subset=["trade_date", "symbol"])


def _select_a_leg_for_date(
    day_signals: pd.DataFrame,
    prev_a_holdings: set[str],
    *,
    theme_quotas: dict[str, int],
    buffer_exit_multiplier: float,
) -> tuple[list[str], dict[str, list[str]]]:
    """Select A-leg holdings for a single date.

    For each theme pool:
    1. Rank stocks within theme by score_a (descending)
    2. New entries: must rank within target quota
    3. Existing holdings: retained until rank > quota * buffer_exit_multiplier
    4. Fill up to quota with best-ranked new stocks

    Args:
        day_signals: Single-date signal DataFrame with score_a, theme columns.
        prev_a_holdings: Set of symbols held in A-leg on previous date.
        theme_quotas: {theme_key: quota_slots}.
        buffer_exit_multiplier: Exit threshold = quota * multiplier.

    Returns:
        (all_selected_symbols, {theme_key: [symbols]})
    """
    selected: list[str] = []
    theme_assignments: dict[str, list[str]] = {}

    for theme_key, quota in theme_quotas.items():
        if quota <= 0:
            continue

        # Get stocks in this theme with valid score_a
        theme_stocks = day_signals[
            (day_signals["theme"] == theme_key) & day_signals["score_a"].notna()
        ].copy()

        if theme_stocks.empty:
            continue

        # Rank by score_a descending
        theme_stocks = theme_stocks.sort_values("score_a", ascending=False)
        symbols_ranked = theme_stocks["symbol"].tolist()

        # Buffer exit threshold
        exit_rank = max(quota + 1, int(quota * buffer_exit_multiplier))

        # Build selection
        pool: list[str] = []
        for rank_idx, symbol in enumerate(symbols_ranked):
            rank_pos = rank_idx + 1  # 1-based rank within theme
            if symbol in prev_a_holdings:
                # Retain if within buffer exit zone
                if rank_pos <= exit_rank:
                    pool.append(symbol)
            else:
                # New entry: only if within target quota
                if rank_pos <= quota:
                    pool.append(symbol)

        # Trim to quota
        pool = pool[:quota]

        # Fill remaining slots with best available new stocks
        if len(pool) < quota:
            for symbol in symbols_ranked:
                if symbol not in pool:
                    pool.append(symbol)
                    if len(pool) >= quota:
                        break

        selected.extend(pool)
        theme_assignments[theme_key] = pool

    return selected, theme_assignments


def _select_b_leg_for_date(
    day_signals: pd.DataFrame,
    prev_b_holdings: set[str],
    *,
    b_slots: int,
    industry_cap: int,
    buffer_exit_rank: int,
    buffer_entry_rank: int,
    max_daily_replacements: int,
) -> list[str]:
    """Select B-leg holdings for a single date.

    Rules:
    1. Rank stocks by score_b descending (full universe, excluding A-leg if needed)
    2. New entries: rank ≤ buffer_entry_rank (default 20)
    3. Existing holdings: retained if rank ≤ buffer_exit_rank (default 35)
    4. Max `industry_cap` stocks per industry
    5. Max `max_daily_replacements` new stocks per day
    6. Fill to exactly b_slots

    Args:
        day_signals: Single-date signal DataFrame with score_b, industry columns.
        prev_b_holdings: Set of B-leg symbols from previous date.
        b_slots: Target number of B-leg slots.
        industry_cap: Max stocks per industry in B-leg.
        buffer_exit_rank: Ranking threshold for retaining existing holdings.
        buffer_entry_rank: Ranking threshold for new entries.
        max_daily_replacements: Maximum new stocks to add in one day.

    Returns:
        List of selected symbols.
    """
    valid = day_signals[day_signals["score_b"].notna()].copy()
    if valid.empty:
        return []

    # Rank by score_b descending
    valid = valid.sort_values("score_b", ascending=False)
    valid["b_rank"] = range(1, len(valid) + 1)

    # Build candidate pool
    selected: list[str] = []
    industry_counts: dict[str, int] = {}
    new_additions = 0

    for _, row in valid.iterrows():
        symbol = str(row["symbol"])
        rank = int(row["b_rank"])
        industry = str(row.get("industry", "")) if pd.notna(row.get("industry")) else ""

        # Check eligibility
        is_held = symbol in prev_b_holdings
        if is_held:
            if rank > buffer_exit_rank:
                continue  # dropped from buffer
        else:
            if rank > buffer_entry_rank:
                continue  # not in entry zone
            if new_additions >= max_daily_replacements:
                continue  # hit daily replacement cap

        # Industry cap
        ind_count = industry_counts.get(industry, 0)
        if ind_count >= industry_cap:
            continue

        # Select
        selected.append(symbol)
        if industry:
            industry_counts[industry] = ind_count + 1
        if not is_held:
            new_additions += 1

        if len(selected) >= b_slots:
            break

    # If we don't have enough, relax constraints
    if len(selected) < b_slots:
        for _, row in valid.iterrows():
            symbol = str(row["symbol"])
            if symbol in selected:
                continue
            rank = int(row["b_rank"])
            if rank > buffer_exit_rank * 2:
                continue
            selected.append(symbol)
            if len(selected) >= b_slots:
                break

    return selected[:b_slots]


def _resolve_overlap(
    a_holdings: list[str],
    b_holdings: list[str],
    *,
    policy: str,
    normal_weight: float,
    max_weight: float,
) -> tuple[list[str], list[str], dict[str, float]]:
    """Resolve A/B overlap and compute final weights.

    Args:
        a_holdings: A-leg symbols.
        b_holdings: B-leg symbols.
        policy: "aggregate" (merge weights) or "deduplicate" (remove from B).
        normal_weight: Weight per slot (default 0.01).
        max_weight: Max weight per stock (default 0.02).

    Returns:
        (a_holdings, b_holdings, {symbol: final_weight})
    """
    a_set = set(a_holdings)
    b_set = set(b_holdings)

    overlap = a_set & b_set

    if policy == "deduplicate":
        b_holdings = [s for s in b_holdings if s not in overlap]
        b_set = set(b_holdings)

    # Build weights
    weights: dict[str, float] = {}
    for symbol in a_set:
        weights[symbol] = normal_weight
    for symbol in b_set:
        if symbol in overlap and policy == "aggregate":
            weights[symbol] = min(max_weight, normal_weight * 2)
        else:
            weights[symbol] = normal_weight

    return list(a_set), list(b_set), weights


def _build_position_rows(
    date_str: str,
    a_holdings: list[str],
    b_holdings: list[str],
    weights: dict[str, float],
    day_signals: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Build position rows for a single rebalance date."""
    rows: list[dict[str, Any]] = []
    signal_lookup = day_signals.set_index("symbol")

    for symbol in set(a_holdings) | set(b_holdings):
        leg = "A" if symbol in a_holdings else "B"
        if symbol in a_holdings and symbol in b_holdings:
            leg = "A+B"

        row: dict[str, Any] = {
            "rebalance_date": date_str,
            "entry_date": date_str,  # daily frequency: same as signal date
            "symbol": symbol,
            "weight": weights.get(symbol, 0.0),
            "side": "long",
            "leg": leg,
        }

        if symbol in signal_lookup.index:
            sig_row = signal_lookup.loc[symbol]
            row["signal"] = float(sig_row.get("score_a", sig_row.get("score_b", np.nan)))
            row["score_a"] = float(sig_row.get("score_a", np.nan))
            row["score_b"] = float(sig_row.get("score_b", np.nan))
            row["theme"] = sig_row.get("theme")
            row["industry"] = sig_row.get("industry")
        else:
            row["signal"] = np.nan
            row["score_a"] = np.nan
            row["score_b"] = np.nan
            row["theme"] = None
            row["industry"] = None

        rows.append(row)

    return rows


def build_style_replica_positions(
    signals: pd.DataFrame,
    *,
    config: StyleReplicaPortfolioConfig | None = None,
) -> pd.DataFrame:
    """Build daily StyleReplica-A80B20 positions from signals.

    Args:
        signals: Long-format signal DataFrame from ``style_replica.signal_generator``.
                 Must contain: signal_date/trade_date, symbol, score_a, score_b,
                 theme, industry, leg.
        config: Portfolio construction configuration.

    Returns:
        DataFrame with columns: rebalance_date, entry_date, symbol, weight,
        signal, rank, side, leg, score_a, score_b, theme, industry.
        Conforms to cstree.positions_by_rebalance contract.
    """
    cfg = config or StyleReplicaPortfolioConfig()
    df = _prepare_signals_frame(signals)
    if df.empty:
        return pd.DataFrame()

    dates = sorted(df["trade_date"].unique())
    all_rows: list[dict[str, Any]] = []
    prev_a: set[str] = set()
    prev_b: set[str] = set()

    for date in dates:
        date_str = pd.Timestamp(date).strftime("%Y%m%d")
        day = df[df["trade_date"] == date]

        if day.empty:
            continue

        # A-leg selection
        a_holdings, _theme_assignments = _select_a_leg_for_date(
            day,
            prev_a,
            theme_quotas=cfg.theme_quotas,
            buffer_exit_multiplier=cfg.a_buffer_exit_multiplier,
        )

        # B-leg selection
        b_holdings = _select_b_leg_for_date(
            day,
            prev_b,
            b_slots=cfg.b_slots,
            industry_cap=cfg.b_industry_cap,
            buffer_exit_rank=cfg.b_buffer_exit_rank,
            buffer_entry_rank=cfg.b_buffer_entry_rank,
            max_daily_replacements=cfg.b_max_daily_replacements,
        )

        # Resolve overlap
        a_final, b_final, weights = _resolve_overlap(
            a_holdings,
            b_holdings,
            policy=cfg.overlap_policy,
            normal_weight=cfg.normal_slot_weight,
            max_weight=cfg.max_name_weight,
        )

        # Ensure size constraints
        while len(a_final) < cfg.a_slots:
            # Try to fill from unselected themed stocks
            themed = day[(day["theme"].notna()) & (day["score_a"].notna())]
            themed = themed[~themed["symbol"].isin(set(a_final))]
            if themed.empty:
                break
            next_stock = themed.sort_values("score_a", ascending=False).iloc[0]["symbol"]
            a_final.append(str(next_stock))

        while len(b_final) < cfg.b_slots:
            remaining = day[(day["score_b"].notna()) & (~day["symbol"].isin(set(b_final)))]
            if remaining.empty:
                break
            next_stock = remaining.sort_values("score_b", ascending=False).iloc[0]["symbol"]
            b_final.append(str(next_stock))

        # Recompute weights after adjustments
        _, _, weights = _resolve_overlap(
            a_final,
            b_final,
            policy=cfg.overlap_policy,
            normal_weight=cfg.normal_slot_weight,
            max_weight=cfg.max_name_weight,
        )

        # Build rows
        rows = _build_position_rows(date_str, a_final, b_final, weights, day)
        all_rows.extend(rows)

        # Update state
        prev_a = set(a_final)
        prev_b = set(b_final)

    if not all_rows:
        return pd.DataFrame()

    positions = pd.DataFrame(all_rows)

    # Add rank within date
    positions["rank"] = (
        positions.groupby("rebalance_date", sort=False)["signal"]
        .rank(ascending=False, method="first", na_option="bottom")
        .astype("Int64")
    )

    # Sort
    positions = positions.sort_values(["rebalance_date", "rank", "symbol"])

    return positions.reset_index(drop=True).reset_index(drop=True)


def compute_daily_changes(
    positions: pd.DataFrame,
) -> pd.DataFrame:
    """Compute day-over-day position changes.

    Args:
        positions: DataFrame from ``build_style_replica_positions``.

    Returns:
        DataFrame with columns: rebalance_date, symbol, action (new/exit/stay),
        leg, weight_change, prev_weight.
    """
    if positions.empty:
        return pd.DataFrame()

    dates = sorted(positions["rebalance_date"].unique())
    changes: list[dict[str, Any]] = []

    for i, date in enumerate(dates):
        day_positions = positions[positions["rebalance_date"] == date]
        current_holdings = dict(zip(day_positions["symbol"], day_positions["weight"], strict=True))
        leg_col = day_positions.get("leg", [None] * len(day_positions))
        current_legs = dict(zip(day_positions["symbol"], leg_col, strict=True))

        prev_holdings: dict[str, float] = {}
        if i > 0:
            prev_day = positions[positions["rebalance_date"] == dates[i - 1]]
            prev_holdings = dict(zip(prev_day["symbol"], prev_day["weight"], strict=True))

        all_symbols = set(current_holdings.keys()) | set(prev_holdings.keys())
        for symbol in sorted(all_symbols):
            cur_w = current_holdings.get(symbol, 0.0)
            prev_w = prev_holdings.get(symbol, 0.0)

            if cur_w > 0 and prev_w == 0:
                action = "new"
            elif cur_w == 0 and prev_w > 0:
                action = "exit"
            elif cur_w != prev_w:
                action = "weight_change"
            else:
                action = "stay"

            changes.append(
                {
                    "rebalance_date": date,
                    "symbol": symbol,
                    "action": action,
                    "leg": current_legs.get(symbol),
                    "weight": cur_w,
                    "prev_weight": prev_w,
                    "weight_change": cur_w - prev_w,
                }
            )

    return pd.DataFrame(changes)


def compute_style_exposure_summary(
    positions: pd.DataFrame,
    *,
    factor_frame: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Compute style exposure summary for a single date's positions.

    Args:
        positions: Single-date position DataFrame.
        factor_frame: Optional wide DataFrame of factor values for the date.

    Returns:
        Dictionary with exposure metrics.
    """
    if positions.empty:
        return {}

    summary: dict[str, Any] = {
        "rebalance_date": str(positions["rebalance_date"].iloc[0]),
        "total_stocks": len(positions),
        "a_leg_count": int((positions["leg"].str.contains("A", na=False)).sum()),
        "b_leg_count": int((positions["leg"].str.contains("B", na=False)).sum()),
        "overlap_count": int((positions["leg"] == "A+B").sum()),
        "total_weight": float(positions["weight"].sum()),
        "a_weight": float(
            positions.loc[positions["leg"].str.contains("A", na=False), "weight"].sum()
        ),
        "b_weight": float(
            positions.loc[positions["leg"].str.contains("B", na=False), "weight"].sum()
        ),
    }

    # Theme distribution
    if "theme" in positions.columns:
        theme_counts = positions["theme"].value_counts().to_dict()
        summary["theme_distribution"] = theme_counts

    # Industry distribution
    if "industry" in positions.columns:
        ind_counts = positions["industry"].value_counts().to_dict()
        summary["industry_distribution"] = ind_counts
        # Max single industry
        if ind_counts:
            max_ind = max(ind_counts, key=ind_counts.get)
            max_ind_pct = ind_counts[max_ind] / len(positions) if len(positions) > 0 else 0
            summary["max_industry_pct"] = round(max_ind_pct, 4)

    return summary


def compute_daily_exposure(
    positions: pd.DataFrame,
) -> pd.DataFrame:
    """Compute daily exposure summaries for the full position history.

    Args:
        positions: Full positions DataFrame from ``build_style_replica_positions``.

    Returns:
        DataFrame with one row per rebalance date, columns for exposure metrics.
    """
    if positions.empty:
        return pd.DataFrame()

    summaries: list[dict[str, Any]] = []
    for date in sorted(positions["rebalance_date"].unique()):
        day = positions[positions["rebalance_date"] == date]
        summaries.append(compute_style_exposure_summary(day))

    return pd.DataFrame(summaries)
