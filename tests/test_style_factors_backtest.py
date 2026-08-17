"""Equivalence / contract tests for the migrated A-share style-factor backtest kernel.

The module ``portfolio_backtester.style_factors_backtest`` holds the
quantile long-short backtest kernel relocated from ``src/style_factors/
factor_backtest.py`` (ADR-0006 R4 slice 7, PR②).  These tests pin the
external behaviour so the migration cannot silently regress the quantile
long-short construction, monthly rebalance scheduling, summary statistics,
and factor-name discovery.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio_backtester.style_factors_backtest import (
    available_factor_names,
    build_factor_returns,
    compute_summary,
    get_rebalance_dates,
)


def _synthetic_frames(
    *,
    days: int = 120,
    symbols: int = 80,
    start: str = "2023-01-03",
    seed: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a deterministic factors frame and a daily-return frame.

    The ``size`` standardized factor is a *slowly drifting* cross-section
    signal: each symbol keeps a persistent latent rank that only drifts a
    little day to day, and its daily return is a stable linear function of
    that persistent rank.  Because the backtest holds a fixed basket between
    rebalances, a persistent (not daily-independent) edge is what lets the
    quantile long-short portfolio earn a sign-stable excess return — which is
    the property these tests pin down.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=days)
    syms = [f"S{i:03d}" for i in range(symbols)]

    # Persistent per-symbol latent rank; high rank => lower future return.
    latent = rng.normal(0.0, 1.0, size=symbols)
    daily_rows: list[dict] = []
    factor_rows: list[dict] = []
    for d in dates:
        latent = latent + rng.normal(0.0, 0.05, size=symbols)  # slow drift
        z = (latent - latent.mean()) / (latent.std() + 1e-9)
        ret = -0.004 * latent + rng.normal(0.0, 0.01, size=symbols)
        for sym, zi, ri in zip(syms, z, ret, strict=True):
            daily_rows.append({"trade_date": d, "symbol": sym, "pct_chg": ri * 100.0})
            factor_rows.append(
                {
                    "trade_date": d,
                    "symbol": sym,
                    "factor_size_z": zi,
                    "factor_value_z": rng.normal(0.0, 1.0),
                }
            )

    daily = pd.DataFrame(daily_rows)
    factors = pd.DataFrame(factor_rows)
    return factors, daily


def test_get_rebalance_dates_is_monthly_last_trading_day() -> None:
    dates = pd.bdate_range("2023-01-01", "2023-04-30")
    rd = get_rebalance_dates(pd.DatetimeIndex(dates))
    assert len(rd) >= 4  # at least one per month
    # Every rebalance date is the last trading day of its month in the input.
    for r in rd:
        same_month = [d for d in dates if d.year == r.year and d.month == r.month]
        assert r == max(same_month)


def test_available_factor_names_ignores_all_nan_columns() -> None:
    factors = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2023-01-03"] * 3),
            "symbol": ["A", "B", "C"],
            "factor_size_z": [0.1, 0.2, 0.3],
            "factor_value_z": [np.nan, np.nan, np.nan],
        }
    )
    names = available_factor_names(factors)
    assert names == ["size"]


def test_build_factor_returns_emits_long_short_for_available_factors() -> None:
    factors, daily = _synthetic_frames()
    rd = get_rebalance_dates(pd.DatetimeIndex(daily["trade_date"].unique()))
    results = build_factor_returns(factors, daily, rd, n_quantiles=5)

    assert "size" in results
    assert "value" in results
    ls = results["size"]["long_short"].dropna()
    assert len(ls) > 0
    # Contract: long-short is a daily return series spanning the rebalance
    # windows, bounded (no silent overflow from the fixed-share compounding
    # path), and both legs are emitted.
    assert ls.abs().max() < 1.0
    assert len(results["size"]["long"].dropna()) > 0
    assert len(results["size"]["short"].dropna()) > 0
    # The synthetic persistent edge is negative: high-size (top quantile) earns
    # less than low-size (bottom quantile), so long-short (high - low) < 0.
    assert results["size"]["long_short"].mean() < 0


def test_compute_summary_reports_sharpe_column() -> None:
    factors, daily = _synthetic_frames()
    rd = get_rebalance_dates(pd.DatetimeIndex(daily["trade_date"].unique()))
    results = build_factor_returns(factors, daily, rd, n_quantiles=5)
    summary = compute_summary(results)

    assert not summary.empty
    assert "sharpe" in summary.columns
    assert set(summary["factor"]) >= {"size", "value"}
    # Summary is sorted by sharpe descending.
    sharpe_vals = summary["sharpe"].to_numpy()
    assert (np.diff(sharpe_vals) <= 1e-9).all()


def test_build_factor_returns_is_deterministic() -> None:
    factors, daily = _synthetic_frames()
    rd = get_rebalance_dates(pd.DatetimeIndex(daily["trade_date"].unique()))

    first = build_factor_returns(factors, daily, rd, n_quantiles=5)
    second = build_factor_returns(factors, daily, rd, n_quantiles=5)

    for name in first:
        assert first[name]["long_short"].equals(second[name]["long_short"])
