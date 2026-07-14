from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from portfolio_backtester._symbol_utils import canonicalize_symbol_columns

_DEFAULT_INDUSTRY_COLUMNS = (
    "industry_name",
    "first_industry_name",
    "second_industry_name",
    "third_industry_name",
    "industry_code",
    "first_industry_code",
    "second_industry_code",
    "third_industry_code",
)

_QUALITY_COLUMNS = (
    "quality",
    "quality_score",
    "roe",
    "roe_ttm",
    "roa",
    "roa_ttm",
    "profit_margin",
    "operating_margin",
    "gross_margin",
    "gross_margin_ttm",
    "cfo_margin",
    "cfo_to_assets",
    "asset_turnover",
)

_MOMENTUM_COLUMNS = (
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

_LOW_VOL_COLUMNS = (
    "low_vol",
    "low_volatility",
    "defensive",
    "rv_120",
    "rv_60",
    "rv_20",
    "volatility_252",
    "volatility_126",
    "volatility_120",
    "volatility_60",
    "volatility_20",
)

_BETA_COLUMNS = (
    "beta",
    "beta_252",
    "beta_126",
    "beta_120",
    "beta_60",
    "market_beta",
)

_STYLE_FACTOR_ORDER = ("size", "value", "quality", "momentum", "low_vol", "beta")
_MISSING_LABEL_TOKENS = frozenset({"", "nan", "none", "<na>", "nat", "null"})


def _empty_style_summary() -> dict[str, Any]:
    return {
        "latest_rebalance_date": None,
        "latest_entry_date": None,
        "factors": {},
        "latest": {},
    }


def _empty_industry_summary() -> dict[str, Any]:
    return {
        "industry_column": None,
        "latest_rebalance_date": None,
        "latest_entry_date": None,
        "latest": {},
    }


def _exposure_period_key(value: object) -> str | None:
    return None if value is None or pd.isna(value) else str(value).strip().removesuffix(".0")


def _build_active_exposure_summary_table(
    style_df: pd.DataFrame,
    industry_df: pd.DataFrame,
    *,
    top_n_industries: int = 3,
) -> pd.DataFrame:
    if style_df.empty and industry_df.empty:
        return pd.DataFrame()

    style_work, industry_work = style_df.copy(), industry_df.copy()
    for frame in (style_work, industry_work):
        if not frame.empty:
            frame["rebalance_date"] = frame["rebalance_date"].map(_exposure_period_key)
            frame["entry_date"] = frame["entry_date"].map(_exposure_period_key)
    periods: set[tuple[str, str | None]] = set()
    for frame in (style_work, industry_work):
        if frame.empty:
            continue
        for _, row in frame[["rebalance_date", "entry_date"]].drop_duplicates().iterrows():
            periods.add((str(row["rebalance_date"]), row["entry_date"]))

    rows: list[dict[str, Any]] = []
    for rebalance_date, entry_date in sorted(periods):
        row: dict[str, Any] = {
            "rebalance_date": rebalance_date,
            "entry_date": entry_date,
        }
        if not style_work.empty:
            style_day = style_work[style_work["rebalance_date"] == rebalance_date]
            for factor in _STYLE_FACTOR_ORDER:
                factor_day = style_day[style_day["factor"] == factor]
                if factor_day.empty:
                    continue
                factor_row = factor_day.iloc[0]
                row[f"{factor}_active_net_vs_equal"] = factor_row["active_net_vs_equal"]
                row[f"{factor}_active_net_vs_cap"] = factor_row["active_net_vs_cap"]
                row[f"{factor}_weight_coverage"] = factor_row["weight_coverage"]
                row[f"{factor}_source"] = factor_row["source"]

        if not industry_work.empty:
            industry_day = industry_work[industry_work["rebalance_date"] == rebalance_date].copy()
            if not industry_day.empty:
                row["industry_column"] = industry_day["industry_col"].dropna().iloc[0]
                reference_col = (
                    "active_net_vs_cap_weight"
                    if industry_day["active_net_vs_cap_weight"].notna().any()
                    else "active_net_vs_equal_weight"
                )
                row["industry_reference"] = reference_col
                ranked = industry_day.assign(
                    abs_active=industry_day[reference_col].abs()
                ).sort_values(["abs_active", "industry"], ascending=[False, True])
                top_industries = ranked.head(top_n_industries).iterrows()
                for idx, (_, ranked_row) in enumerate(top_industries, start=1):
                    row[f"industry_top_{idx}_name"] = ranked_row["industry"]
                    row[f"industry_top_{idx}_active"] = ranked_row[reference_col]
                    row[f"industry_top_{idx}_portfolio_net_weight"] = ranked_row[
                        "portfolio_net_weight"
                    ]

        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_df.sort_values("rebalance_date", inplace=True)
    summary_df.reset_index(drop=True, inplace=True)
    return summary_df


def _to_datetime_series(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, format="%Y%m%d", errors="coerce")
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(values.loc[missing], errors="coerce")
    return parsed.dt.normalize()


def _clean_categorical_labels(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series(dtype="object", index=series.index)
    values = series.astype("string").str.strip()
    values = values.mask(values.str.lower().isin(_MISSING_LABEL_TOKENS))
    return values.astype("object")


def _as_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _zscore(series: pd.Series) -> pd.Series:
    values = _as_numeric(series)
    mask = values.notna()
    if int(mask.sum()) < 2:
        return pd.Series(np.nan, index=series.index, dtype=float)
    mean = float(values.loc[mask].mean())
    std = float(values.loc[mask].std(ddof=0))
    if not np.isfinite(std) or std <= 0:
        return pd.Series(np.nan, index=series.index, dtype=float)
    return (values - mean) / std


def _safe_log(series: pd.Series) -> pd.Series:
    values = _as_numeric(series)
    values = values.where(values > 0)
    return np.log(values)


def _resolve_industry_column(
    frame: pd.DataFrame,
    industry_columns: Sequence[str] | None = None,
) -> str | None:
    candidates = list(dict.fromkeys(list(industry_columns or []) + list(_DEFAULT_INDUSTRY_COLUMNS)))
    for column in candidates:
        if column not in frame.columns:
            continue
        cleaned = _clean_categorical_labels(frame[column]).dropna()
        if cleaned.empty:
            continue
        return column
    return None


def _weighted_average(values: pd.Series, weights: pd.Series) -> float:
    aligned = pd.concat(
        [values.rename("value"), pd.to_numeric(weights, errors="coerce").rename("weight")],
        axis=1,
    ).dropna()
    if aligned.empty:
        return np.nan
    total = float(aligned["weight"].sum())
    if not np.isfinite(total) or total <= 0:
        return np.nan
    return float((aligned["value"] * aligned["weight"]).sum() / total)


def _update_factor_meta(
    factor_meta: dict[str, dict[str, Any]],
    factor: str,
    meta: Mapping[str, Any],
) -> None:
    existing = factor_meta.get(factor)
    if existing is None:
        factor_meta[factor] = dict(meta)
        return
    if bool(existing.get("available")):
        return
    if bool(meta.get("available")):
        factor_meta[factor] = dict(meta)


def _price_history_tables(
    pricing_data: pd.DataFrame | None,
    *,
    price_col: str,
) -> dict[str, Any]:
    if pricing_data is None or pricing_data.empty or price_col not in pricing_data.columns:
        return {
            "price_table": pd.DataFrame(),
            "returns": pd.DataFrame(),
            "momentum_tables": {},
            "vol_tables": {},
        }
    work = pricing_data.copy()
    work = canonicalize_symbol_columns(work, context="Exposure pricing data")
    work["trade_date"] = pd.to_datetime(work["trade_date"], errors="coerce").dt.normalize()
    work = work.dropna(subset=["trade_date", "symbol"])
    work = work.drop_duplicates(subset=["trade_date", "symbol"], keep="last")
    price_table = (
        work.pivot(index="trade_date", columns="symbol", values=price_col)
        .sort_index()
        .apply(pd.to_numeric, errors="coerce")
    )
    returns = price_table.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    momentum_tables = {
        window: price_table.pct_change(window, fill_method=None).replace(
            [np.inf, -np.inf],
            np.nan,
        )
        for window in (60, 120, 252)
    }
    vol_tables = {
        window: returns.rolling(window=window, min_periods=max(20, window // 2)).std(ddof=0)
        for window in (20, 60, 120)
    }
    return {
        "price_table": price_table,
        "returns": returns,
        "momentum_tables": momentum_tables,
        "vol_tables": vol_tables,
    }


def _build_benchmark_daily_returns(
    benchmark_df: pd.DataFrame | None,
    benchmark_return_series: pd.Series | None,
    *,
    price_col: str,
) -> pd.Series:
    if benchmark_return_series is not None and not benchmark_return_series.empty:
        series = benchmark_return_series.copy()
        series.index = pd.to_datetime(series.index, errors="coerce").normalize()
        series = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
        return series.dropna().sort_index()
    if benchmark_df is None or benchmark_df.empty or price_col not in benchmark_df.columns:
        return pd.Series(dtype=float, name="benchmark_return")
    work = benchmark_df.copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"], errors="coerce").dt.normalize()
    work = work.dropna(subset=["trade_date"])
    prices = (
        work.sort_values("trade_date")
        .drop_duplicates(subset=["trade_date"], keep="last")
        .set_index("trade_date")[price_col]
    )
    returns = pd.to_numeric(prices, errors="coerce").pct_change().replace([np.inf, -np.inf], np.nan)
    returns.name = "benchmark_return"
    return returns.dropna().sort_index()


def _build_beta_table(
    daily_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
) -> pd.DataFrame:
    if daily_returns.empty or benchmark_returns.empty:
        return pd.DataFrame(index=daily_returns.index, columns=daily_returns.columns, dtype=float)
    benchmark = benchmark_returns.reindex(daily_returns.index).astype(float)
    mean_stock = daily_returns.rolling(window=120, min_periods=60).mean()
    mean_bench = benchmark.rolling(window=120, min_periods=60).mean()
    mean_prod = daily_returns.mul(benchmark, axis=0).rolling(window=120, min_periods=60).mean()
    cov = mean_prod.sub(mean_stock.mul(mean_bench, axis=0), axis=0)
    var = benchmark.rolling(window=120, min_periods=60).var(ddof=0)
    beta = cov.div(var, axis=0)
    return beta.replace([np.inf, -np.inf], np.nan)


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


def _industry_exposure_rows(
    *,
    positions: pd.DataFrame,
    day: pd.DataFrame,
    industry_col: str,
    market_cap_col: str | None,
    rebalance_date: pd.Timestamp,
    entry_date: pd.Timestamp | None,
) -> list[dict[str, Any]]:
    universe = day[["symbol", industry_col]].drop_duplicates(subset=["symbol"], keep="last").copy()
    universe[industry_col] = _clean_categorical_labels(universe[industry_col])
    universe = universe.dropna(subset=[industry_col]).copy()
    if universe.empty:
        return []

    pos = positions.groupby("symbol", as_index=False)["weight"].sum()
    pos = pos.merge(universe, on="symbol", how="inner")

    long = pos.loc[pos["weight"] > 0].copy()
    short = pos.loc[pos["weight"] < 0].copy()
    short["weight"] = short["weight"].abs()
    gross = pos.copy()
    gross["weight"] = gross["weight"].abs()

    long_total = float(long["weight"].sum()) if not long.empty else 0.0
    short_total = float(short["weight"].sum()) if not short.empty else 0.0
    gross_total = float(gross["weight"].sum()) if not gross.empty else 0.0

    long_share = (
        long.groupby(industry_col)["weight"].sum() / long_total
        if long_total > 0
        else pd.Series(dtype=float)
    )
    short_share = (
        short.groupby(industry_col)["weight"].sum() / short_total
        if short_total > 0
        else pd.Series(dtype=float)
    )
    gross_share = (
        gross.groupby(industry_col)["weight"].sum() / gross_total
        if gross_total > 0
        else pd.Series(dtype=float)
    )
    universe_equal = universe.groupby(industry_col)["symbol"].nunique()
    universe_equal = (
        universe_equal / float(universe_equal.sum()) if not universe_equal.empty else universe_equal
    )

    universe_cap = pd.Series(dtype=float)
    if market_cap_col and market_cap_col in day.columns:
        cap = (
            day[["symbol", industry_col, market_cap_col]]
            .drop_duplicates(subset=["symbol"], keep="last")
            .copy()
        )
        cap[market_cap_col] = _as_numeric(cap[market_cap_col])
        cap = cap.loc[cap[market_cap_col] > 0]
        cap[industry_col] = _clean_categorical_labels(cap[industry_col])
        cap = cap.dropna(subset=[industry_col])
        if not cap.empty:
            universe_cap = cap.groupby(industry_col)[market_cap_col].sum()
            universe_cap = universe_cap / float(universe_cap.sum())

    industries = sorted(
        set(universe_equal.index)
        | set(long_share.index)
        | set(short_share.index)
        | set(gross_share.index)
        | set(universe_cap.index)
    )
    rows: list[dict[str, Any]] = []
    for industry in industries:
        long_weight = float(long_share.get(industry, 0.0))
        short_weight = float(short_share.get(industry, 0.0))
        gross_weight = float(gross_share.get(industry, 0.0))
        net_weight = float(long_weight - short_weight)
        equal_weight = float(universe_equal.get(industry, 0.0))
        cap_weight = float(universe_cap.get(industry, np.nan))
        rows.append(
            {
                "rebalance_date": rebalance_date.strftime("%Y%m%d"),
                "entry_date": entry_date.strftime("%Y%m%d") if entry_date is not None else None,
                "industry": str(industry),
                "industry_col": industry_col,
                "portfolio_long_weight": long_weight,
                "portfolio_short_weight": short_weight,
                "portfolio_net_weight": net_weight,
                "portfolio_gross_weight": gross_weight,
                "universe_equal_weight": equal_weight,
                "universe_cap_weight": cap_weight,
                "active_net_vs_equal_weight": float(net_weight - equal_weight),
                "active_net_vs_cap_weight": (
                    float(net_weight - cap_weight) if np.isfinite(cap_weight) else np.nan
                ),
            }
        )
    return rows


def _build_industry_history(
    frame: pd.DataFrame,
    *,
    industry_col: str,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    history = frame[["symbol", "trade_date", industry_col]].copy()
    history[industry_col] = _clean_categorical_labels(history[industry_col])
    history = history.dropna(subset=["symbol", "trade_date", industry_col])
    if history.empty:
        return {}
    history = history.sort_values(["symbol", "trade_date"]).drop_duplicates(
        subset=["symbol", "trade_date"],
        keep="last",
    )

    by_symbol: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for symbol, group in history.groupby("symbol", sort=False):
        dates = group["trade_date"].to_numpy(dtype="datetime64[ns]")
        labels = group[industry_col].to_numpy(dtype=object)
        if len(dates) == 0:
            continue
        by_symbol[str(symbol)] = (dates, labels)
    return by_symbol


def _apply_industry_labels_asof(
    day: pd.DataFrame,
    *,
    industry_col: str,
    rebalance_date: pd.Timestamp,
    industry_history: Mapping[str, tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    work = day.copy()
    if industry_col in work.columns:
        industry_values = _clean_categorical_labels(work[industry_col])
    else:
        industry_values = pd.Series(pd.NA, index=work.index, dtype="object")

    missing = industry_values.isna()
    if missing.any():
        rebalance_dt64 = rebalance_date.to_datetime64()
        for idx, symbol in work.loc[missing, "symbol"].items():
            history = industry_history.get(str(symbol))
            if history is None:
                continue
            dates, labels = history
            pos = int(np.searchsorted(dates, rebalance_dt64, side="right") - 1)
            if pos >= 0:
                industry_values.at[idx] = labels[pos]

    work[industry_col] = industry_values
    return work


def _empty_exposure_result() -> dict[str, Any]:
    return {
        "style": pd.DataFrame(),
        "style_summary": _empty_style_summary(),
        "industry": pd.DataFrame(),
        "industry_summary": _empty_industry_summary(),
        "active_summary": pd.DataFrame(),
    }


def _normalize_exposure_scored_data(scored_data: pd.DataFrame) -> pd.DataFrame:
    scored = scored_data.copy()
    scored = canonicalize_symbol_columns(scored, context="Exposure scored data")
    scored["trade_date"] = pd.to_datetime(scored["trade_date"], errors="coerce").dt.normalize()
    scored = scored.dropna(subset=["trade_date", "symbol"])
    return scored.drop_duplicates(subset=["trade_date", "symbol"], keep="last")


def _normalize_exposure_industry_source(
    industry_source_data: pd.DataFrame | None,
) -> pd.DataFrame:
    if industry_source_data is None or industry_source_data.empty:
        return pd.DataFrame()
    industry_source = industry_source_data.copy()
    industry_source = canonicalize_symbol_columns(
        industry_source,
        context="Exposure industry source",
    )
    industry_source["trade_date"] = pd.to_datetime(
        industry_source["trade_date"], errors="coerce"
    ).dt.normalize()
    industry_source = industry_source.dropna(subset=["trade_date", "symbol"])
    return industry_source.drop_duplicates(subset=["trade_date", "symbol"], keep="last")


def _normalize_exposure_positions(positions_by_rebalance: pd.DataFrame) -> pd.DataFrame:
    positions = positions_by_rebalance.copy()
    positions = canonicalize_symbol_columns(positions, context="Exposure positions")
    positions["rebalance_date_ts"] = _to_datetime_series(positions["rebalance_date"])
    positions["entry_date_ts"] = _to_datetime_series(positions["entry_date"])
    return positions.dropna(subset=["rebalance_date_ts", "entry_date_ts", "symbol"])


def _resolve_exposure_industry_context(
    scored: pd.DataFrame,
    industry_source: pd.DataFrame,
    *,
    industry_columns: Sequence[str] | None,
) -> tuple[str | None, Mapping[str, tuple[np.ndarray, np.ndarray]]]:
    industry_col = _resolve_industry_column(scored, industry_columns=industry_columns)
    if industry_col is None and not industry_source.empty:
        industry_col = _resolve_industry_column(
            industry_source,
            industry_columns=industry_columns,
        )
    industry_history_source = (
        industry_source
        if industry_col is not None and industry_col in industry_source.columns
        else scored
    )
    industry_history = (
        _build_industry_history(industry_history_source, industry_col=industry_col)
        if industry_col is not None
        else {}
    )
    return industry_col, industry_history


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


def _build_exposure_rows(
    *,
    scored: pd.DataFrame,
    positions: pd.DataFrame,
    history: Mapping[str, Any],
    beta_table: pd.DataFrame,
    industry_col: str | None,
    industry_history: Mapping[str, tuple[np.ndarray, np.ndarray]],
    market_cap_col: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    style_rows: list[dict[str, Any]] = []
    industry_rows: list[dict[str, Any]] = []
    factor_meta: dict[str, dict[str, Any]] = {}
    by_date = scored.groupby("trade_date", sort=True)
    for rebalance_date, pos_day in positions.groupby("rebalance_date_ts", sort=True):
        if rebalance_date not in by_date.groups:
            continue
        day = by_date.get_group(rebalance_date).copy()
        entry_date = pos_day["entry_date_ts"].iloc[0] if not pos_day.empty else None
        style_rows.extend(
            _style_rows_for_rebalance(
                day=day,
                positions=pos_day,
                rebalance_date=rebalance_date,
                entry_date=entry_date,
                market_cap_col=market_cap_col,
                history=history,
                beta_table=beta_table,
                factor_meta=factor_meta,
            )
        )
        if industry_col is not None:
            industry_day = _apply_industry_labels_asof(
                day,
                industry_col=industry_col,
                rebalance_date=rebalance_date,
                industry_history=industry_history,
            )
            industry_rows.extend(
                _industry_exposure_rows(
                    positions=pos_day,
                    day=industry_day,
                    industry_col=industry_col,
                    market_cap_col=market_cap_col,
                    rebalance_date=rebalance_date,
                    entry_date=entry_date,
                )
            )
    return style_rows, industry_rows, factor_meta


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


def _finalize_industry_exposure(
    industry_rows: list[dict[str, Any]],
    industry_col: str | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    industry_df = pd.DataFrame(industry_rows)
    if not industry_df.empty:
        industry_df.sort_values(["rebalance_date", "industry"], inplace=True)
        industry_df.reset_index(drop=True, inplace=True)
    industry_summary = _empty_industry_summary()
    industry_summary["industry_column"] = industry_col
    if industry_df.empty:
        return industry_df, industry_summary
    latest_rebalance = str(industry_df["rebalance_date"].max())
    latest_industry = industry_df[industry_df["rebalance_date"] == latest_rebalance].copy()
    industry_summary["latest_rebalance_date"] = latest_rebalance
    latest_entry = latest_industry["entry_date"].dropna()
    industry_summary["latest_entry_date"] = (
        str(latest_entry.iloc[0]) if not latest_entry.empty else None
    )
    reference_col = (
        "active_net_vs_cap_weight"
        if latest_industry["active_net_vs_cap_weight"].notna().any()
        else "active_net_vs_equal_weight"
    )
    latest_industry["abs_active"] = latest_industry[reference_col].abs()
    latest_industry = latest_industry.sort_values("abs_active", ascending=False)
    industry_summary["latest"] = {
        "reference": reference_col,
        "top_absolute_active": [
            {
                "industry": str(row["industry"]),
                "portfolio_net_weight": float(row["portfolio_net_weight"]),
                "universe_equal_weight": float(row["universe_equal_weight"]),
                "universe_cap_weight": (
                    float(row["universe_cap_weight"])
                    if pd.notna(row["universe_cap_weight"])
                    else np.nan
                ),
                "active_net_vs_equal_weight": float(row["active_net_vs_equal_weight"]),
                "active_net_vs_cap_weight": (
                    float(row["active_net_vs_cap_weight"])
                    if pd.notna(row["active_net_vs_cap_weight"])
                    else np.nan
                ),
            }
            for _, row in latest_industry.head(10).iterrows()
        ],
    }
    return industry_df, industry_summary


def compute_backtest_exposure_analysis(
    scored_data: pd.DataFrame | None,
    positions_by_rebalance: pd.DataFrame | None,
    *,
    pricing_data: pd.DataFrame | None = None,
    price_col: str = "close",
    benchmark_df: pd.DataFrame | None = None,
    benchmark_return_series: pd.Series | None = None,
    market_cap_col: str | None = None,
    industry_columns: Sequence[str] | None = None,
    industry_source_data: pd.DataFrame | None = None,
) -> dict[str, Any]:
    if (
        scored_data is None
        or scored_data.empty
        or positions_by_rebalance is None
        or positions_by_rebalance.empty
    ):
        return _empty_exposure_result()

    scored = _normalize_exposure_scored_data(scored_data)
    industry_source = _normalize_exposure_industry_source(industry_source_data)
    positions = _normalize_exposure_positions(positions_by_rebalance)
    if positions.empty:
        return _empty_exposure_result()

    history = _price_history_tables(pricing_data, price_col=price_col)
    benchmark_returns = _build_benchmark_daily_returns(
        benchmark_df,
        benchmark_return_series,
        price_col=price_col,
    )
    beta_table = _build_beta_table(history["returns"], benchmark_returns)
    industry_col, industry_history = _resolve_exposure_industry_context(
        scored,
        industry_source,
        industry_columns=industry_columns,
    )
    style_rows, industry_rows, factor_meta = _build_exposure_rows(
        scored=scored,
        positions=positions,
        history=history,
        beta_table=beta_table,
        industry_col=industry_col,
        industry_history=industry_history,
        market_cap_col=market_cap_col,
    )
    style_df, style_summary = _finalize_style_exposure(style_rows, factor_meta)
    industry_df, industry_summary = _finalize_industry_exposure(industry_rows, industry_col)
    return {
        "style": style_df,
        "style_summary": style_summary,
        "industry": industry_df,
        "industry_summary": industry_summary,
        "active_summary": _build_active_exposure_summary_table(style_df, industry_df),
    }
