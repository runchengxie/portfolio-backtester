"""Factor portfolio backtest — quintile long-short, summary, correlations, yearly breakdown."""

from __future__ import annotations

import numpy as np
import pandas as pd

FACTOR_NAMES = [
    "size",
    "value",
    "momentum",
    "quality",
    "earnings_yield",
    "lowvol",
    "growth",
    "leverage",
    "beta",
    "liquidity",
    # New factors from locally-landed tushare datasets (zero network traffic):
    "liquidity_flow",
    "chip_concentration",
    "institution_holding",
    "dividend_yield",
    "ps_value",
]


def available_factor_names(factors_df: pd.DataFrame) -> list[str]:
    """Return factors whose standardized columns exist in the factor frame."""
    names = []
    for name in FACTOR_NAMES:
        column = f"factor_{name}_z"
        if column in factors_df.columns and factors_df[column].notna().any():
            names.append(name)
    return names


def get_rebalance_dates(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Monthly rebalance: last trading day of each month."""
    df = pd.DataFrame({"date": dates})
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    return pd.DatetimeIndex(df.groupby(["year", "month"])["date"].max().sort_values())


def _daily_return_matrix(daily: pd.DataFrame) -> pd.DataFrame:
    ret_df = daily[["trade_date", "symbol", "pct_chg"]].dropna().copy()
    ret_df["trade_date"] = pd.to_datetime(ret_df["trade_date"]).dt.normalize()
    ret_df["pct_chg"] = ret_df["pct_chg"] / 100.0
    returns = ret_df.pivot_table(
        index="trade_date",
        columns="symbol",
        values="pct_chg",
        aggfunc="mean",
    )
    returns.index = pd.DatetimeIndex(returns.index)
    return returns.sort_index()


def _buy_and_hold_leg_returns(
    period_returns: pd.DataFrame,
    symbols: list[str],
) -> pd.Series:
    """Return an equal-weight leg whose shares are fixed until rebalance.

    Missing marks inside the holding window are treated as zero return so a
    suspended name keeps its capital weight instead of silently reallocating it
    to the remaining names.  Delisting terminal returns are still unavailable in
    the current raw input and remain an explicit research limitation.
    """
    if not symbols or period_returns.empty:
        return pd.Series(dtype=float, index=period_returns.index)

    returns = period_returns.reindex(columns=symbols).fillna(0.0)
    weights = np.full(len(symbols), 1.0 / len(symbols), dtype=float)
    portfolio_returns: list[float] = []
    for row in returns.to_numpy(dtype=float):
        portfolio_return = float(weights @ row)
        portfolio_returns.append(portfolio_return)
        gross_weights = weights * (1.0 + row)
        gross_total = float(gross_weights.sum())
        if gross_total > 0:
            weights = gross_weights / gross_total
    return pd.Series(portfolio_returns, index=returns.index, dtype=float)


def _concat_return_parts(parts: list[pd.Series], *, name: str) -> pd.Series:
    if not parts:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([], name="trade_date"), name=name)
    result = pd.concat(parts).sort_index()
    result.name = name
    return result


def _formation_quantiles(
    factors_df: pd.DataFrame,
    *,
    trade_date: pd.Timestamp,
    signal_column: str,
    n_quantiles: int,
) -> pd.DataFrame | None:
    formation = factors_df[factors_df["trade_date"] == trade_date].dropna(subset=[signal_column])
    if len(formation) < n_quantiles * 10:
        return None
    formation = formation.sort_values(signal_column).copy()
    formation["quantile"] = pd.qcut(
        formation[signal_column], n_quantiles, labels=False, duplicates="drop"
    )
    return formation if formation["quantile"].nunique() == n_quantiles else None


def _resolve_requested_quantiles(
    n_quantiles: int,
    requested_quantiles: tuple[int, ...] | None,
) -> tuple[int, ...]:
    if n_quantiles < 2:
        raise ValueError("n_quantiles must be at least 2")
    quantiles = requested_quantiles or tuple(range(1, n_quantiles + 1))
    if not quantiles or any(value < 1 or value > n_quantiles for value in quantiles):
        raise ValueError("requested_quantiles must be within the configured quantile range")
    return quantiles


def _build_signal_quantile_result(
    factors_df: pd.DataFrame,
    daily_returns: pd.DataFrame,
    rebalance_dates: list[pd.Timestamp],
    *,
    signal_name: str,
    signal_column: str,
    n_quantiles: int,
    quantiles: tuple[int, ...],
    include_universe: bool,
) -> dict[str, object]:
    quantile_parts: dict[int, list[pd.Series]] = {value: [] for value in quantiles}
    universe_parts: list[pd.Series] = []

    for index, rebalance_date in enumerate(rebalance_dates[:-1]):
        next_rebalance_date = rebalance_dates[index + 1]
        rebalance_date = pd.Timestamp(rebalance_date).normalize()  # type: ignore
        next_rebalance_date = pd.Timestamp(next_rebalance_date).normalize()  # type: ignore
        formation = _formation_quantiles(
            factors_df,
            trade_date=rebalance_date,
            signal_column=signal_column,
            n_quantiles=n_quantiles,
        )
        if formation is None:
            continue

        period_returns = daily_returns.loc[rebalance_date:next_rebalance_date]
        period_returns = period_returns[period_returns.index > rebalance_date]
        if period_returns.empty:
            continue

        for quantile in quantiles:
            symbols = formation[formation["quantile"] == quantile - 1]["symbol"].tolist()
            quantile_parts[quantile].append(_buy_and_hold_leg_returns(period_returns, symbols))
        if include_universe:
            universe_parts.append(
                _buy_and_hold_leg_returns(period_returns, formation["symbol"].tolist())
            )

    quantile_returns = {
        quantile: _concat_return_parts(parts, name=f"{signal_name}_q{quantile}")
        for quantile, parts in quantile_parts.items()
    }
    low = quantile_returns.get(1, pd.Series(dtype=float))
    high = quantile_returns.get(n_quantiles, pd.Series(dtype=float))
    paired = pd.concat({"high": high, "low": low}, axis=1).dropna()
    long_short = paired["high"] - paired["low"] if not paired.empty else pd.Series(dtype=float)
    long_short.name = signal_name
    universe = _concat_return_parts(universe_parts, name=f"{signal_name}_universe")
    long_excess_pair = pd.concat({"long": high, "universe": universe}, axis=1).dropna()
    long_excess = (
        long_excess_pair["long"] - long_excess_pair["universe"]
        if not long_excess_pair.empty
        else pd.Series(dtype=float)
    )
    long_excess.name = f"{signal_name}_long_excess"
    return {
        "quantiles": quantile_returns,
        "long": high,
        "short": low,
        "long_short": long_short,
        "universe": universe,
        "long_excess": long_excess,
    }


def build_quantile_portfolio_returns(
    factors_df: pd.DataFrame,
    daily: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    signal_columns: dict[str, str],
    *,
    n_quantiles: int = 5,
    requested_quantiles: tuple[int, ...] | None = None,
    include_universe: bool = True,
) -> dict[str, dict[str, object]]:
    """Build fixed-share quantile portfolios for arbitrary formation-date signals.

    Quantile 1 contains the lowest signal scores and ``n_quantiles`` contains
    the highest.  The function is shared by the standard factor backtest and
    focused diagnostics that need every quantile and an eligible-universe
    benchmark.
    """
    quantiles = _resolve_requested_quantiles(n_quantiles, requested_quantiles)

    daily_returns = _daily_return_matrix(daily)
    rd_list = sorted(rebalance_dates)

    results: dict[str, dict[str, object]] = {}
    for signal_name, signal_column in signal_columns.items():
        if signal_column not in factors_df.columns:
            continue
        print(f"[backtest] {signal_name} ...", flush=True)
        results[signal_name] = _build_signal_quantile_result(
            factors_df,
            daily_returns,
            rd_list,
            signal_name=signal_name,
            signal_column=signal_column,
            n_quantiles=n_quantiles,
            quantiles=quantiles,
            include_universe=include_universe,
        )

    return results


def build_factor_returns(
    factors_df: pd.DataFrame,
    daily: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    n_quantiles: int = 5,
) -> dict:
    """For each factor: quintile long-short monthly rebalance."""
    factor_names = available_factor_names(factors_df)
    signal_columns = {name: f"factor_{name}_z" for name in factor_names}
    detailed = build_quantile_portfolio_returns(
        factors_df,
        daily,
        rebalance_dates,
        signal_columns,
        n_quantiles=n_quantiles,
        requested_quantiles=(1, n_quantiles),
        include_universe=False,
    )
    return {
        name: {
            "long_short": result["long_short"],
            "long": result["long"],
            "short": result["short"],
        }
        for name, result in detailed.items()
    }


def _max_drawdown(returns: pd.Series) -> float:
    cumulative = (1 + returns).cumprod()
    drawdown = (cumulative - cumulative.cummax()) / cumulative.cummax()
    return float(drawdown.min())


def compute_summary(factor_results: dict, trading_days: int = 252) -> pd.DataFrame:
    rows = []
    for name, res in factor_results.items():
        ls = res["long_short"].dropna()
        if len(ls) < 20:
            continue
        daily_mean = ls.mean()
        cumulative_ret = (1 + ls).prod() - 1
        annual_ret = (1 + daily_mean) ** trading_days - 1
        geometric_annual_ret = (1 + cumulative_ret) ** (trading_days / len(ls)) - 1
        annual_vol = ls.std() * np.sqrt(trading_days)
        sharpe = daily_mean / ls.std() * np.sqrt(trading_days) if ls.std() > 0 else 0
        hit_rate = (ls > 0).mean()
        n_years = (ls.index.max() - ls.index.min()).days / 365.25
        rows.append(
            {
                "factor": name,
                "days": len(ls),
                "years": round(n_years, 1),
                "cumulative_ret": round(cumulative_ret * 100, 2),
                "annual_ret": round(annual_ret * 100, 2),
                "geometric_annual_ret": round(geometric_annual_ret * 100, 2),
                "annual_vol": round(annual_vol * 100, 2),
                "sharpe": round(sharpe, 2),
                "max_drawdown": round(_max_drawdown(ls) * 100, 2),
                "hit_rate": round(hit_rate * 100, 1),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "factor",
                "days",
                "years",
                "cumulative_ret",
                "annual_ret",
                "geometric_annual_ret",
                "annual_vol",
                "sharpe",
                "max_drawdown",
                "hit_rate",
            ]
        )
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False)


def compute_factor_correlations(factor_results: dict) -> pd.DataFrame:
    series = {}
    for name, res in factor_results.items():
        s = res["long_short"].dropna()
        if len(s) > 20:
            series[name] = s
    return pd.DataFrame(series).corr()


def compute_yearly_breakdown(factor_results: dict) -> pd.DataFrame:
    rows = []
    for name, res in factor_results.items():
        s = res["long_short"].dropna()
        for year_end, group in s.resample("YE"):
            if len(group) < 50:
                continue
            annual_ret = (1 + group).prod() - 1
            ann_vol = group.std() * np.sqrt(252)
            sharpe = group.mean() / group.std() * np.sqrt(252) if group.std() > 0 else 0
            rows.append(
                {
                    "year": year_end.year,
                    "factor": name,
                    "days": len(group),
                    "period_start": group.index.min().date().isoformat(),
                    "period_end": group.index.max().date().isoformat(),
                    "is_partial_year": bool(
                        group.index.min().month > 1 or group.index.max().month < 12
                    ),
                    "period_return": round(annual_ret * 100, 2),
                    "annual_ret": round(annual_ret * 100, 2),
                    "annual_vol": round(ann_vol * 100, 2),
                    "sharpe": round(sharpe, 2),
                    "max_drawdown": round(_max_drawdown(group) * 100, 2),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "year",
                "factor",
                "days",
                "period_start",
                "period_end",
                "is_partial_year",
                "period_return",
                "annual_ret",
                "annual_vol",
                "sharpe",
                "max_drawdown",
            ]
        )
    return pd.DataFrame(rows).sort_values(["year", "factor"])
