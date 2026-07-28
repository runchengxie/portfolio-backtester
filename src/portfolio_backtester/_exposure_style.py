from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from ._exposure_columns import (
    _BETA_COLUMNS,
    _LOW_VOL_COLUMNS,
    _MOMENTUM_COLUMNS,
    _QUALITY_COLUMNS,
    _as_numeric,
    _empty_style_summary,
    _safe_log,
    _update_factor_meta,
    _weighted_average,
    _zscore,
)


def _compose_from_columns(
    day: pd.DataFrame,
    specs: Sequence[tuple[str, str]],
) -> tuple[pd.Series | None, list[str]]:
    components: list[pd.Series] = []
    used: list[str] = []
    for column, transform in specs:
        if column not in day.columns:
            continue
        values = _as_numeric(day[column])
        if transform == "identity":
            transformed = values
        elif transform == "neg":
            transformed = -values
        elif transform == "log":
            transformed = _safe_log(day[column])
        elif transform == "neg_log":
            transformed = -_safe_log(day[column])
        else:
            continue
        if int(transformed.notna().sum()) == 0:
            continue
        components.append(transformed)
        used.append(column)
    if not components:
        return None, []
    if len(components) == 1:
        return components[0], used
    return pd.concat(components, axis=1).mean(axis=1, skipna=True), used


def _resolve_size_factor(
    day: pd.DataFrame,
    *,
    market_cap_col: str | None,
) -> tuple[pd.Series | None, dict[str, Any]]:
    candidates: list[tuple[str, str]] = []
    if market_cap_col:
        candidates.append((market_cap_col, "log" if market_cap_col != "log_mcap" else "identity"))
    candidates.extend(
        [
            ("log_mcap", "identity"),
            ("market_cap", "log"),
            ("hk_total_market_val", "log"),
        ]
    )
    values, used = _compose_from_columns(day, candidates)
    return values, {
        "available": values is not None,
        "source": "columns",
        "columns": used,
    }


def _resolve_value_factor(day: pd.DataFrame) -> tuple[pd.Series | None, dict[str, Any]]:
    values, used = _compose_from_columns(
        day,
        [
            ("value", "identity"),
            ("value_score", "identity"),
            ("bp", "identity"),
            ("book_to_price", "identity"),
            ("pb", "neg_log"),
            ("pb_ratio_ttm", "neg_log"),
            ("pe_ttm", "neg_log"),
            ("pe_ratio_ttm", "neg_log"),
        ],
    )
    return values, {
        "available": values is not None,
        "source": "columns",
        "columns": used,
    }


def _resolve_quality_factor(day: pd.DataFrame) -> tuple[pd.Series | None, dict[str, Any]]:
    values, used = _compose_from_columns(day, [(column, "identity") for column in _QUALITY_COLUMNS])
    return values, {
        "available": values is not None,
        "source": "columns",
        "columns": used,
    }


def _resolve_momentum_factor(
    day: pd.DataFrame,
    rebalance_date: pd.Timestamp,
    *,
    momentum_tables: Mapping[int, pd.DataFrame],
) -> tuple[pd.Series | None, dict[str, Any]]:
    values, used = _compose_from_columns(
        day,
        [(column, "identity") for column in _MOMENTUM_COLUMNS],
    )
    if values is not None:
        return values, {
            "available": True,
            "source": "columns",
            "columns": used,
        }

    derived_components: list[pd.Series] = []
    derived_labels: list[str] = []
    for window, table in momentum_tables.items():
        if rebalance_date not in table.index:
            continue
        row = _as_numeric(table.loc[rebalance_date].reindex(day["symbol"]))
        if int(row.notna().sum()) == 0:
            continue
        derived_components.append(
            pd.Series(row.to_numpy(dtype=float), index=day.index, dtype=float)
        )
        derived_labels.append(f"price_return_{window}d")
    if not derived_components:
        return None, {
            "available": False,
            "source": None,
            "columns": [],
        }
    if len(derived_components) == 1:
        values = derived_components[0]
    else:
        values = pd.concat(derived_components, axis=1).mean(axis=1, skipna=True)
    return values, {
        "available": True,
        "source": "derived_price_history",
        "columns": derived_labels,
    }


def _resolve_low_vol_factor(
    day: pd.DataFrame,
    rebalance_date: pd.Timestamp,
    *,
    vol_tables: Mapping[int, pd.DataFrame],
) -> tuple[pd.Series | None, dict[str, Any]]:
    direct_specs = [
        ("low_vol", "identity"),
        ("low_volatility", "identity"),
        ("defensive", "identity"),
    ]
    inverse_specs = [
        (column, "neg")
        for column in _LOW_VOL_COLUMNS
        if column not in {"low_vol", "low_volatility", "defensive"}
    ]
    values, used = _compose_from_columns(day, [*direct_specs, *inverse_specs])
    if values is not None:
        return values, {
            "available": True,
            "source": "columns",
            "columns": used,
        }

    derived_components: list[pd.Series] = []
    derived_labels: list[str] = []
    for window, table in vol_tables.items():
        if rebalance_date not in table.index:
            continue
        row = -_as_numeric(table.loc[rebalance_date].reindex(day["symbol"]))
        if int(row.notna().sum()) == 0:
            continue
        derived_components.append(
            pd.Series(row.to_numpy(dtype=float), index=day.index, dtype=float)
        )
        derived_labels.append(f"realized_vol_{window}d")
    if not derived_components:
        return None, {
            "available": False,
            "source": None,
            "columns": [],
        }
    if len(derived_components) == 1:
        values = derived_components[0]
    else:
        values = pd.concat(derived_components, axis=1).mean(axis=1, skipna=True)
    return values, {
        "available": True,
        "source": "derived_price_history",
        "columns": derived_labels,
    }


def _resolve_beta_factor(
    day: pd.DataFrame,
    rebalance_date: pd.Timestamp,
    *,
    beta_table: pd.DataFrame,
) -> tuple[pd.Series | None, dict[str, Any]]:
    values, used = _compose_from_columns(day, [(column, "identity") for column in _BETA_COLUMNS])
    if values is not None:
        return values, {
            "available": True,
            "source": "columns",
            "columns": used,
        }
    if beta_table.empty or rebalance_date not in beta_table.index:
        return None, {
            "available": False,
            "source": None,
            "columns": [],
        }
    row = _as_numeric(beta_table.loc[rebalance_date].reindex(day["symbol"]))
    if int(row.notna().sum()) == 0:
        return None, {
            "available": False,
            "source": None,
            "columns": [],
        }
    return pd.Series(row.to_numpy(dtype=float), index=day.index, dtype=float), {
        "available": True,
        "source": "derived_price_history",
        "columns": ["rolling_beta_120d"],
    }


def _style_exposure_base_fields(
    factor: str,
    *,
    positions: pd.DataFrame,
    rebalance_date: pd.Timestamp,
    entry_date: pd.Timestamp | None,
    source_meta: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "rebalance_date": rebalance_date.strftime("%Y%m%d"),
        "entry_date": entry_date.strftime("%Y%m%d") if entry_date is not None else None,
        "factor": factor,
        "source": source_meta.get("source"),
        "source_columns": list(source_meta.get("columns") or []),
        "n_universe": 0,
        "n_holdings": int(positions["symbol"].nunique()) if not positions.empty else 0,
        "weight_coverage": 0.0,
    }


def _empty_style_exposure_row(base_fields: dict[str, Any]) -> dict[str, Any]:
    return {
        **base_fields,
        "portfolio_long": np.nan,
        "portfolio_short": np.nan,
        "portfolio_net": np.nan,
        "portfolio_gross": np.nan,
        "universe_equal": np.nan,
        "universe_cap_weight": np.nan,
        "active_net_vs_equal": np.nan,
        "active_net_vs_cap": np.nan,
    }


def _aligned_style_day(values: pd.Series, day: pd.DataFrame) -> pd.DataFrame:
    aligned_day = day.loc[values.notna(), ["symbol"]].copy()
    aligned_day["factor_value"] = values.loc[values.notna()].to_numpy(dtype=float)
    if aligned_day.empty:
        return aligned_day

    z = _zscore(aligned_day["factor_value"])
    aligned_day["factor_z"] = z.to_numpy(dtype=float)
    return aligned_day.dropna(subset=["factor_z"]).drop_duplicates(
        subset=["symbol"],
        keep="last",
    )


def _portfolio_net_style_exposure(portfolio_long: float, portfolio_short: float) -> float:
    if np.isfinite(portfolio_long) and np.isfinite(portfolio_short):
        return float(portfolio_long - portfolio_short)
    if np.isfinite(portfolio_long):
        return float(portfolio_long)
    if np.isfinite(portfolio_short):
        return float(-portfolio_short)
    return np.nan


def _portfolio_style_stats(positions: pd.DataFrame, z_by_symbol: pd.Series) -> dict[str, float]:
    positions_work = positions.groupby("symbol", as_index=False)["weight"].sum()
    positions_work = positions_work.merge(
        z_by_symbol.rename("factor_z"),
        left_on="symbol",
        right_index=True,
        how="left",
    )
    positions_work = positions_work.dropna(subset=["factor_z"])

    total_abs = float(positions["weight"].abs().sum()) if not positions.empty else 0.0
    covered_abs = float(positions_work["weight"].abs().sum()) if not positions_work.empty else 0.0
    weight_coverage = covered_abs / total_abs if total_abs > 0 else 0.0

    long_weights = positions_work.loc[positions_work["weight"] > 0, ["factor_z", "weight"]]
    short_weights = positions_work.loc[positions_work["weight"] < 0, ["factor_z", "weight"]].copy()
    short_weights["weight"] = short_weights["weight"].abs()
    gross_weights = positions_work[["factor_z", "weight"]].copy()
    gross_weights["weight"] = gross_weights["weight"].abs()

    portfolio_long = _weighted_average(long_weights["factor_z"], long_weights["weight"])
    portfolio_short = _weighted_average(short_weights["factor_z"], short_weights["weight"])
    return {
        "weight_coverage": float(weight_coverage),
        "portfolio_long": portfolio_long,
        "portfolio_short": portfolio_short,
        "portfolio_net": _portfolio_net_style_exposure(portfolio_long, portfolio_short),
        "portfolio_gross": _weighted_average(gross_weights["factor_z"], gross_weights["weight"]),
    }


def _cap_weighted_style_universe(
    *,
    day: pd.DataFrame,
    market_cap_col: str | None,
    z_by_symbol: pd.Series,
) -> float:
    if not market_cap_col or market_cap_col not in day.columns:
        return np.nan

    cap_weights = day[["symbol", market_cap_col]].drop_duplicates(subset=["symbol"], keep="last")
    cap_weights[market_cap_col] = _as_numeric(cap_weights[market_cap_col])
    cap_weights = cap_weights.loc[cap_weights[market_cap_col] > 0]
    if cap_weights.empty:
        return np.nan

    cap_weights = cap_weights.merge(
        z_by_symbol.rename("factor_z"),
        left_on="symbol",
        right_index=True,
        how="inner",
    )
    return _weighted_average(cap_weights["factor_z"], cap_weights[market_cap_col])


def _style_active_delta(portfolio_net: float, universe_reference: float) -> float:
    if np.isfinite(portfolio_net) and np.isfinite(universe_reference):
        return float(portfolio_net - universe_reference)
    return np.nan


def _style_exposure_row(
    factor: str,
    *,
    values: pd.Series,
    positions: pd.DataFrame,
    day: pd.DataFrame,
    market_cap_col: str | None,
    rebalance_date: pd.Timestamp,
    entry_date: pd.Timestamp | None,
    source_meta: Mapping[str, Any],
) -> dict[str, Any]:
    base_fields = _style_exposure_base_fields(
        factor,
        positions=positions,
        rebalance_date=rebalance_date,
        entry_date=entry_date,
        source_meta=source_meta,
    )
    aligned_day = _aligned_style_day(values, day)
    if aligned_day.empty:
        return _empty_style_exposure_row(base_fields)

    z_by_symbol = aligned_day.set_index("symbol")["factor_z"]
    portfolio = _portfolio_style_stats(positions, z_by_symbol)
    portfolio_net = portfolio["portfolio_net"]
    universe_equal = float(aligned_day["factor_z"].mean()) if not aligned_day.empty else np.nan
    universe_cap_weight = _cap_weighted_style_universe(
        day=day,
        market_cap_col=market_cap_col,
        z_by_symbol=z_by_symbol,
    )

    return {
        **base_fields,
        "n_universe": int(aligned_day["symbol"].nunique()),
        "weight_coverage": portfolio["weight_coverage"],
        "portfolio_long": portfolio["portfolio_long"],
        "portfolio_short": portfolio["portfolio_short"],
        "portfolio_net": portfolio["portfolio_net"],
        "portfolio_gross": portfolio["portfolio_gross"],
        "universe_equal": universe_equal,
        "universe_cap_weight": universe_cap_weight,
        "active_net_vs_equal": _style_active_delta(portfolio_net, universe_equal),
        "active_net_vs_cap": _style_active_delta(portfolio_net, universe_cap_weight),
    }


def _style_rows_for_rebalance(
    *,
    day: pd.DataFrame,
    positions: pd.DataFrame,
    rebalance_date: pd.Timestamp,
    entry_date: pd.Timestamp | None,
    market_cap_col: str | None,
    history: Mapping[str, Any],
    beta_table: pd.DataFrame,
    factor_meta: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    factor_specs = [
        ("size", *_resolve_size_factor(day, market_cap_col=market_cap_col)),
        ("value", *_resolve_value_factor(day)),
        ("quality", *_resolve_quality_factor(day)),
        (
            "momentum",
            *_resolve_momentum_factor(
                day,
                rebalance_date,
                momentum_tables=history["momentum_tables"],
            ),
        ),
        (
            "low_vol",
            *_resolve_low_vol_factor(
                day,
                rebalance_date,
                vol_tables=history["vol_tables"],
            ),
        ),
        ("beta", *_resolve_beta_factor(day, rebalance_date, beta_table=beta_table)),
    ]
    rows: list[dict[str, Any]] = []
    for factor, values, source_meta in factor_specs:
        _update_factor_meta(factor_meta, factor, source_meta)
        rows.append(
            _style_exposure_row(
                factor,
                values=values if values is not None else pd.Series(np.nan, index=day.index),
                positions=positions,
                day=day,
                market_cap_col=market_cap_col,
                rebalance_date=rebalance_date,
                entry_date=entry_date,
                source_meta=source_meta,
            )
        )
    return rows


def _finalize_style_exposure(
    style_rows: list[dict[str, Any]],
    factor_meta: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    style_df = pd.DataFrame(style_rows)
    if not style_df.empty:
        style_df.sort_values(["rebalance_date", "factor"], inplace=True)
        style_df.reset_index(drop=True, inplace=True)
    style_summary = _empty_style_summary()
    style_summary["factors"] = factor_meta
    if style_df.empty:
        return style_df, style_summary
    latest_rebalance = str(style_df["rebalance_date"].max())
    latest_style = style_df[style_df["rebalance_date"] == latest_rebalance]
    style_summary["latest_rebalance_date"] = latest_rebalance
    latest_entry = latest_style["entry_date"].dropna()
    style_summary["latest_entry_date"] = (
        str(latest_entry.iloc[0]) if not latest_entry.empty else None
    )
    style_summary["latest"] = {
        str(row["factor"]): {
            "portfolio_long": float(row["portfolio_long"])
            if pd.notna(row["portfolio_long"])
            else np.nan,
            "portfolio_short": float(row["portfolio_short"])
            if pd.notna(row["portfolio_short"])
            else np.nan,
            "portfolio_net": float(row["portfolio_net"])
            if pd.notna(row["portfolio_net"])
            else np.nan,
            "universe_equal": float(row["universe_equal"])
            if pd.notna(row["universe_equal"])
            else np.nan,
            "universe_cap_weight": (
                float(row["universe_cap_weight"])
                if pd.notna(row["universe_cap_weight"])
                else np.nan
            ),
            "active_net_vs_equal": (
                float(row["active_net_vs_equal"])
                if pd.notna(row["active_net_vs_equal"])
                else np.nan
            ),
            "active_net_vs_cap": (
                float(row["active_net_vs_cap"]) if pd.notna(row["active_net_vs_cap"]) else np.nan
            ),
            "source": row["source"],
            "source_columns": list(row["source_columns"])
            if isinstance(row["source_columns"], list)
            else [],
            "weight_coverage": float(row["weight_coverage"]),
        }
        for _, row in latest_style.iterrows()
    }
    return style_df, style_summary
