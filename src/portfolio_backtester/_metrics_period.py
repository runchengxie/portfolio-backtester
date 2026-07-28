"""Period-return summary metrics (annualized, risk-adjusted, drawdown timing)."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd


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
