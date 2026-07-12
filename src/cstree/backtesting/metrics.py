from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from cstree.backtesting._symbol_utils import canonicalize_symbol_columns

from .portfolio_selection import apply_rank_offset, apply_rebalance_buffer

try:
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover - optional dependency
    scipy_stats = None

__all__ = [
    "daily_ic_series",
    "estimate_turnover",
    "quantile_returns",
    "summarize_active_returns",
    "summarize_ic",
    "summarize_period_returns",
]


def spearman_corr(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 2:
        return np.nan
    return x.rank(method="average").corr(y.rank(method="average"))


def pearson_corr(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 2:
        return np.nan
    return x.corr(y)


def daily_ic_series(
    data: pd.DataFrame,
    target_col: str,
    pred_col: str,
    *,
    method: str = "spearman",
) -> pd.Series:
    method = str(method).strip().lower()
    if method == "spearman":
        corr_fn = spearman_corr
    elif method == "pearson":
        corr_fn = pearson_corr
    else:
        raise ValueError("method must be one of: spearman, pearson.")

    records: list[tuple[pd.Timestamp, float]] = []
    for date, group in data.groupby("trade_date"):
        if group[target_col].nunique() < 2:
            continue
        ic = corr_fn(group[pred_col], group[target_col])
        if not np.isnan(ic):
            records.append((pd.to_datetime(date), float(ic)))
    if not records:
        return pd.Series(dtype=float, name="ic")
    records.sort(key=lambda x: x[0])
    return pd.Series(
        [value for _, value in records],
        index=pd.Index([date for date, _ in records], name="trade_date"),
        name="ic",
    )


def summarize_ic(ic_series: pd.Series) -> dict[str, float]:
    if ic_series is None or ic_series.empty:
        return _empty_ic_summary()
    values = ic_series.dropna()
    n = int(values.shape[0])
    if n == 0:
        return _empty_ic_summary()

    mean = float(values.mean())
    std = float(values.std(ddof=0))
    ir = mean / std if std > 0 else np.nan
    t_stat = mean / (std / np.sqrt(n)) if std > 0 else np.nan
    p_value = np.nan
    if scipy_stats is not None and np.isfinite(t_stat) and n > 1:
        p_value = float(2 * scipy_stats.t.sf(abs(t_stat), df=n - 1))
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "ir": ir,
        "t_stat": t_stat,
        "p_value": p_value,
    }


def _empty_ic_summary() -> dict[str, float]:
    return {
        "n": 0,
        "mean": np.nan,
        "std": np.nan,
        "ir": np.nan,
        "t_stat": np.nan,
        "p_value": np.nan,
    }


def quantile_returns(
    data: pd.DataFrame,
    pred_col: str,
    target_col: str,
    n_quantiles: int,
) -> pd.DataFrame:
    def _add_quantile(values: pd.Series) -> pd.Series:
        if len(values) < n_quantiles:
            return pd.Series([np.nan] * len(values), index=values.index)
        ranks = values.rank(method="first")
        return pd.qcut(ranks, n_quantiles, labels=False)

    data = data.copy()
    quantile = data.groupby("trade_date")[pred_col].apply(_add_quantile)
    data["quantile"] = quantile.reset_index(level=0, drop=True)
    data = data.dropna(subset=["quantile"])

    q_ret = data.groupby(["trade_date", "quantile"])[target_col].mean().unstack()
    q_ret.index = pd.to_datetime(q_ret.index)
    return q_ret


def estimate_turnover(
    data: pd.DataFrame,
    pred_col: str,
    k: int,
    rebalance_dates: list[pd.Timestamp],
    buffer_exit: int = 0,
    buffer_entry: int = 0,
    rank_offset: int = 0,
) -> pd.Series:
    if data is None or data.empty:
        return pd.Series(dtype=float, name="turnover")
    data = canonicalize_symbol_columns(data, context="Turnover data")
    prev = None
    turnovers: list[tuple[pd.Timestamp, float]] = []
    day_groups = {  # noqa: C416 - avoid relying on a shadowable dict() callable here.
        date: group for date, group in data.groupby("trade_date", sort=False)
    }
    for date in rebalance_dates:
        day = day_groups.get(date)
        if day is None or len(day) < k:
            continue
        ranked = apply_rank_offset(
            day.sort_values(pred_col, ascending=False)["symbol"].tolist(),
            rank_offset,
        )
        k_final = min(k, len(ranked))
        if k_final <= 0:
            continue
        holdings = set(
            apply_rebalance_buffer(
                ranked,
                prev,
                k_final,
                buffer_exit,
                buffer_entry,
            )[:k_final]
        )
        if prev is not None:
            overlap = len(holdings & prev)
            turnovers.append((pd.to_datetime(date), 1 - overlap / k_final))
        prev = holdings
    if not turnovers:
        return pd.Series(dtype=float, name="turnover")
    turnovers.sort(key=lambda x: x[0])
    return pd.Series(
        [value for _, value in turnovers],
        index=pd.Index([date for date, _ in turnovers], name="trade_date"),
        name="turnover",
    )


def summarize_active_returns(
    strategy: pd.Series,
    benchmark: pd.Series,
    periods_per_year: float,
) -> tuple[dict[str, float], pd.Series]:
    aligned = pd.concat(
        [strategy.rename("strategy"), benchmark.rename("benchmark")], axis=1
    ).dropna()
    if aligned.empty:
        return _empty_active_summary(), pd.Series(dtype=float, name="active_return")

    strategy = aligned["strategy"]
    benchmark = aligned["benchmark"]
    active = strategy - benchmark
    mean = float(active.mean())
    std = float(active.std(ddof=1)) if active.shape[0] > 1 else np.nan
    tracking_error, information_ratio = _tracking_stats(mean, std, periods_per_year)
    beta = _beta(strategy, benchmark)
    alpha = (
        float((strategy.mean() - beta * benchmark.mean()) * periods_per_year)
        if np.isfinite(beta) and np.isfinite(periods_per_year)
        else np.nan
    )
    corr = float(strategy.corr(benchmark)) if strategy.shape[0] > 1 else np.nan
    active_total = _active_total_return(strategy, benchmark)

    return {
        "n": int(active.shape[0]),
        "mean": mean,
        "std": std,
        "tracking_error": tracking_error,
        "information_ratio": information_ratio,
        "beta": beta,
        "alpha": alpha,
        "corr": corr,
        "active_total_return": active_total,
    }, active.rename("active_return")


def _empty_active_summary() -> dict[str, float]:
    return {
        "n": 0,
        "mean": np.nan,
        "std": np.nan,
        "tracking_error": np.nan,
        "information_ratio": np.nan,
        "beta": np.nan,
        "alpha": np.nan,
        "corr": np.nan,
        "active_total_return": np.nan,
    }


def _tracking_stats(mean: float, std: float, periods_per_year: float) -> tuple[float, float]:
    if np.isfinite(std) and std > 0 and np.isfinite(periods_per_year):
        return std * np.sqrt(periods_per_year), mean / std * np.sqrt(periods_per_year)
    return np.nan, np.nan


def _beta(strategy: pd.Series, benchmark: pd.Series) -> float:
    bench_var = float(benchmark.var(ddof=1)) if benchmark.shape[0] > 1 else np.nan
    if np.isfinite(bench_var) and bench_var > 0:
        return float(strategy.cov(benchmark) / bench_var)
    return np.nan


def _active_total_return(strategy: pd.Series, benchmark: pd.Series) -> float:
    strat_total = float((1 + strategy).prod() - 1.0)
    bench_total = float((1 + benchmark).prod() - 1.0)
    if np.isfinite(strat_total) and np.isfinite(bench_total):
        active_total = (1 + strat_total) / (1 + bench_total) - 1.0
        return float(active_total) if np.isfinite(active_total) else np.nan
    return np.nan


def _drawdown_timing(nav: pd.Series) -> dict[str, float]:
    if nav is None:
        return _empty_drawdown_timing()
    nav = nav.dropna()
    if nav.empty:
        return _empty_drawdown_timing()

    values = nav.to_numpy(dtype=float)
    running_max = np.maximum.accumulate(values)
    drawdown = values / running_max - 1.0
    trough_pos = int(np.nanargmin(drawdown))
    peak_value = float(running_max[trough_pos])
    pre_peak = values[: trough_pos + 1]
    peak_candidates = np.flatnonzero(np.isclose(pre_peak, peak_value))
    peak_pos = 0 if peak_candidates.size == 0 else int(peak_candidates[-1])
    drawdown_duration = float(trough_pos - peak_pos)
    recovery_time, recovery_days = _recovery_timing(nav, values, trough_pos, peak_value)
    drawdown_days = _index_day_distance(nav, trough_pos, peak_pos)

    return {
        "drawdown_duration": drawdown_duration,
        "recovery_time": recovery_time,
        "drawdown_duration_days": drawdown_days,
        "recovery_time_days": recovery_days,
    }


def _empty_drawdown_timing() -> dict[str, float]:
    return {
        "drawdown_duration": np.nan,
        "recovery_time": np.nan,
        "drawdown_duration_days": np.nan,
        "recovery_time_days": np.nan,
    }


def _recovery_timing(
    nav: pd.Series,
    values: np.ndarray,
    trough_pos: int,
    peak_value: float,
) -> tuple[float, float]:
    post_nav = values[trough_pos:]
    recovery_candidates = np.flatnonzero(post_nav >= peak_value)
    if recovery_candidates.size == 0:
        return np.nan, np.nan
    recovery_pos = trough_pos + int(recovery_candidates[0])
    recovery_time = float(recovery_pos - trough_pos)
    return recovery_time, _index_day_distance(nav, recovery_pos, trough_pos)


def _index_day_distance(nav: pd.Series, end_pos: int, start_pos: int) -> float:
    if isinstance(nav.index, pd.DatetimeIndex):
        dt_index = cast(pd.DatetimeIndex, nav.index)
        return float((dt_index[end_pos] - dt_index[start_pos]).days)
    return np.nan


def _empty_period_return_summary() -> dict:
    return {
        "periods": 0,
        "total_return": np.nan,
        "ann_return": np.nan,
        "ann_vol": np.nan,
        "sharpe": np.nan,
        "max_drawdown": np.nan,
        "avg_holding": np.nan,
        "periods_per_year": np.nan,
        "sortino": np.nan,
        "calmar": np.nan,
        "drawdown_duration": np.nan,
        "recovery_time": np.nan,
        "drawdown_duration_days": np.nan,
        "recovery_time_days": np.nan,
        "skew": np.nan,
        "kurtosis": np.nan,
        "var_95": np.nan,
        "cvar_95": np.nan,
        "avg_exit_lag_days": np.nan,
        "max_exit_lag_days": np.nan,
        "periods_with_delayed_exit": 0,
        "delayed_exit_ratio": np.nan,
    }


def _annualized_return(
    total_return: float,
    period_info: list[dict],
    trading_days_per_year: int,
) -> float:
    total_days = np.nan
    if period_info:
        entry_first = period_info[0]["entry_idx"]
        exit_last = period_info[-1]["exit_idx"]
        total_days = exit_last - entry_first
    if np.isfinite(total_days) and total_days > 0:
        return (1 + total_return) ** (trading_days_per_year / total_days) - 1.0
    return np.nan


def _holding_period_stats(
    period_info: list[dict],
    trading_days_per_year: int,
) -> tuple[float, float]:
    holding_lengths = [info["exit_idx"] - info["entry_idx"] for info in period_info]
    avg_holding = float(np.mean(holding_lengths)) if holding_lengths else np.nan
    periods_per_year = (
        float(trading_days_per_year / avg_holding)
        if np.isfinite(avg_holding) and avg_holding > 0
        else np.nan
    )
    return avg_holding, periods_per_year


def _risk_adjusted_stats(
    returns: pd.Series,
    periods_per_year: float,
    max_drawdown: float,
    ann_return: float,
) -> tuple[float, float, float, float]:
    period_vol = returns.std(ddof=1)
    if np.isfinite(period_vol) and period_vol > 0 and np.isfinite(periods_per_year):
        ann_vol = period_vol * np.sqrt(periods_per_year)
        sharpe = returns.mean() / period_vol * np.sqrt(periods_per_year)
    else:
        ann_vol = np.nan
        sharpe = np.nan

    downside = np.minimum(returns.to_numpy(), 0.0)
    downside_std = float(np.sqrt(np.mean(downside**2))) if len(downside) > 0 else np.nan
    if np.isfinite(downside_std) and downside_std > 0 and np.isfinite(periods_per_year):
        sortino = float(returns.mean() / downside_std * np.sqrt(periods_per_year))
    else:
        sortino = np.nan

    calmar = (
        float(ann_return / abs(max_drawdown))
        if np.isfinite(max_drawdown) and max_drawdown < 0 and np.isfinite(ann_return)
        else np.nan
    )
    return ann_vol, sharpe, sortino, calmar


def _period_exit_lag(info: dict) -> float | None:
    lag_raw = info.get("exit_delay_steps")
    if lag_raw is None:
        planned_idx = info.get("planned_exit_idx")
        exit_idx = info.get("exit_idx")
        if planned_idx is not None and exit_idx is not None:
            lag_raw = int(exit_idx) - int(planned_idx)
    if lag_raw is None:
        return None
    try:
        lag = float(lag_raw)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(lag):
        return None
    return max(0.0, lag)


def _exit_lag_stats(period_info: list[dict]) -> tuple[float, float, int, float]:
    exit_lags = [lag for info in period_info if (lag := _period_exit_lag(info)) is not None]
    if not exit_lags:
        return np.nan, np.nan, 0, np.nan

    avg_exit_lag = float(np.mean(exit_lags))
    max_exit_lag = float(np.max(exit_lags))
    delayed_periods = int(sum(lag > 0 for lag in exit_lags))
    delayed_ratio = delayed_periods / float(len(exit_lags))
    return avg_exit_lag, max_exit_lag, delayed_periods, delayed_ratio


def _distribution_stats(returns: pd.Series) -> tuple[float, float, float, float]:
    skew = float(returns.skew()) if returns.shape[0] > 2 else np.nan
    kurtosis = float(returns.kurtosis()) if returns.shape[0] > 3 else np.nan
    if returns.shape[0] > 0:
        var_95 = float(np.nanpercentile(returns, 5))
        tail = cast(pd.Series, returns[returns <= var_95])
        cvar_95 = float(tail.mean()) if not tail.empty else np.nan
    else:
        var_95 = np.nan
        cvar_95 = np.nan
    return skew, kurtosis, var_95, cvar_95


def summarize_period_returns(
    returns: pd.Series,
    period_info: list[dict],
    trading_days_per_year: int,
) -> dict:
    if returns is None or returns.empty:
        return _empty_period_return_summary()

    nav = (1 + returns).cumprod()
    total_return = nav.iloc[-1] - 1.0
    max_drawdown = (nav / nav.cummax() - 1.0).min()
    ann_return = _annualized_return(total_return, period_info, trading_days_per_year)
    avg_holding, periods_per_year = _holding_period_stats(period_info, trading_days_per_year)
    ann_vol, sharpe, sortino, calmar = _risk_adjusted_stats(
        returns,
        periods_per_year,
        max_drawdown,
        ann_return,
    )
    timing = _drawdown_timing(nav)
    avg_exit_lag, max_exit_lag, delayed_periods, delayed_ratio = _exit_lag_stats(period_info)
    skew, kurtosis, var_95, cvar_95 = _distribution_stats(returns)

    return {
        "periods": len(returns),
        "total_return": total_return,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "avg_holding": avg_holding,
        "periods_per_year": periods_per_year,
        "sortino": sortino,
        "calmar": calmar,
        **timing,
        "skew": skew,
        "kurtosis": kurtosis,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "avg_exit_lag_days": avg_exit_lag,
        "max_exit_lag_days": max_exit_lag,
        "periods_with_delayed_exit": delayed_periods,
        "delayed_exit_ratio": delayed_ratio,
        **_calendar_half_year_split(returns),
    }


def _calendar_half_year_split(returns: pd.Series) -> dict:
    """Compute H1/H2 calendar-year split metrics on period returns.

    Returns a flat dict with keys like 'h1_return', 'h2_return',
    'h1h2_gap', etc.  If the series spans only one half-year the
    missing half is filled with None.
    """
    if returns.empty:
        return {"h1_return": None, "h2_return": None, "h1h2_gap": None}
    idx = pd.DatetimeIndex(returns.index)
    h1_mask = idx.month <= 6
    h2_mask = idx.month >= 7
    h1_ret = float(returns.loc[h1_mask].sum()) if h1_mask.any() else None
    h2_ret = float(returns.loc[h2_mask].sum()) if h2_mask.any() else None
    gap = (h1_ret - h2_ret) if (h1_ret is not None and h2_ret is not None) else None
    return {"h1_return": h1_ret, "h2_return": h2_ret, "h1h2_gap": gap}
