"""Information-coefficient metrics for signal evaluation."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd

try:
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover - optional dependency
    scipy_stats: Any = None


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
            records.append((pd.to_datetime(cast(Any, date)), float(ic)))
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


def leg_attribution_frame(
    data: pd.DataFrame,
    pred_col: str,
    target_col: str,
    *,
    top_quantile: float = 0.10,
    bottom_quantile: float = 0.10,
) -> pd.DataFrame:
    """Per-date top and bottom leg returns, spread and excess decomposition.

    Each row is one cross-section. The top leg keeps the highest
    ``top_quantile`` fraction of the signal, the bottom leg keeps the lowest
    ``bottom_quantile`` fraction. ``top_excess`` is the top leg return minus
    the cross-sectional mean return, ``bottom_drag`` is the mean return minus
    the bottom leg return, and ``spread`` is their sum.

    Returns columns: top_ret, bottom_ret, cross_mean, spread, top_excess,
    bottom_drag.
    """
    records: list[dict[str, float | pd.Timestamp]] = []
    for raw_date, group in data.groupby("trade_date"):
        values = group[[pred_col, target_col]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(values) < 2:
            continue
        ranked = values.sort_values(pred_col, ascending=False)
        n_high = max(int(np.ceil(len(ranked) * top_quantile)), 1)
        n_low = max(int(np.ceil(len(ranked) * bottom_quantile)), 1)
        top = ranked.head(n_high)
        bottom = ranked.tail(n_low)
        cross_mean = float(values[target_col].mean())
        top_ret = float(top[target_col].mean())
        bottom_ret = float(bottom[target_col].mean())
        records.append(
            {
                "trade_date": pd.to_datetime(cast(Any, raw_date)),
                "top_ret": top_ret,
                "bottom_ret": bottom_ret,
                "cross_mean": cross_mean,
                "spread": top_ret - bottom_ret,
                "top_excess": top_ret - cross_mean,
                "bottom_drag": cross_mean - bottom_ret,
            }
        )
    if not records:
        return pd.DataFrame(
            columns=["top_ret", "bottom_ret", "cross_mean", "spread", "top_excess", "bottom_drag"]
        )
    frame = pd.DataFrame(records).set_index("trade_date")
    frame.index = pd.to_datetime(frame.index)
    frame.index.name = "trade_date"
    return frame


def summarize_leg_attribution(
    frame: pd.DataFrame,
    *,
    period: str = "M",
) -> pd.DataFrame:
    """Aggregate leg attribution over a period.

    ``period`` is passed to pandas ``to_period``, for example "M" for month,
    "Q" for quarter, "Y" for year. Returns top_excess, bottom_drag and the
    bottom share of the spread.

    Returns columns: period, n_dates, top_ret, bottom_ret, cross_mean,
    spread, top_excess, bottom_drag, bottom_share.
    """
    if frame is None or frame.empty:
        return pd.DataFrame(
            columns=[
                "period",
                "n_dates",
                "top_ret",
                "bottom_ret",
                "cross_mean",
                "spread",
                "top_excess",
                "bottom_drag",
                "bottom_share",
            ]
        )
    index = pd.DatetimeIndex(frame.index)
    per = index.to_period(period)
    grouped = frame.copy()
    grouped["period"] = per
    agg = (
        grouped.groupby("period")
        .agg(
            n_dates=("top_ret", "size"),
            top_ret=("top_ret", "mean"),
            bottom_ret=("bottom_ret", "mean"),
            cross_mean=("cross_mean", "mean"),
            spread=("spread", "mean"),
            top_excess=("top_excess", "mean"),
            bottom_drag=("bottom_drag", "mean"),
        )
        .reset_index()
    )
    spread = agg["spread"].replace(0.0, np.nan)
    agg["bottom_share"] = agg["bottom_drag"] / spread
    return agg
