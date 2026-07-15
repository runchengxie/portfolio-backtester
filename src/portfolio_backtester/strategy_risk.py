"""Strategy-level risk and backtest diagnostics inspired by AFML."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt

import numpy as np
import pandas as pd

from .sharpe_inference import (
    probabilistic_sharpe_ratio as probabilistic_sharpe_ratio_from_stats,
)


@dataclass(frozen=True)
class StrategyRiskReport:
    observations: int
    periods_per_year: float
    sharpe: float
    probabilistic_sharpe: float
    positive_return_hhi: float
    negative_return_hhi: float
    time_concentration_hhi: float
    hit_ratio: float
    average_hit: float
    average_miss: float
    bets_per_year: float
    strategy_failure_probability: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def probabilistic_sharpe_ratio(
    returns: pd.Series,
    *,
    benchmark_sharpe: float = 0.0,
    periods_per_year: float = 252.0,
) -> float:
    """Estimate the probability that the true Sharpe exceeds a benchmark."""

    values = pd.to_numeric(returns, errors="coerce").dropna()
    n = len(values)
    if n < 3:
        return float("nan")
    std = float(values.std(ddof=1))
    if not np.isfinite(std) or std <= 0:
        return float("nan")
    if not np.isfinite(periods_per_year) or periods_per_year <= 0:
        return float("nan")
    observed_periodic = float(values.mean()) / std
    benchmark_periodic = float(benchmark_sharpe) / sqrt(periods_per_year)
    return probabilistic_sharpe_ratio_from_stats(
        sharpe=observed_periodic,
        benchmark_sharpe=benchmark_periodic,
        periods=n,
        skew=float(values.skew()),
        kurtosis_excess=float(values.kurtosis()),
    )


def normalized_hhi(values: pd.Series) -> float:
    """Return a sample-size adjusted HHI in [0, 1]."""

    clean = pd.to_numeric(values, errors="coerce").dropna().abs()
    n = len(clean)
    total = float(clean.sum())
    if n <= 1 or total <= 0:
        return float("nan")
    raw = float(np.square(clean / total).sum())
    return float((raw - 1.0 / n) / (1.0 - 1.0 / n))


def return_concentration(
    returns: pd.Series,
    *,
    time_frequency: str = "M",
) -> dict[str, float]:
    """Calculate positive, negative, and time concentration of bet returns."""

    values = pd.to_numeric(returns, errors="coerce").dropna()
    positive = normalized_hhi(values.loc[values > 0])
    negative = normalized_hhi(values.loc[values < 0])
    if isinstance(values.index, pd.DatetimeIndex):
        periods = values.index.to_period(time_frequency)
        time_values = values.abs().groupby(periods).sum()
        time_hhi = normalized_hhi(time_values.loc[time_values > 0])
    else:
        time_hhi = float("nan")
    return {
        "positive_return_hhi": positive,
        "negative_return_hhi": negative,
        "time_concentration_hhi": time_hhi,
    }


def implementation_shortfall_metrics(
    *,
    gross_returns: pd.Series,
    net_returns: pd.Series,
    turnover: pd.Series,
    execution_costs: pd.Series | None = None,
) -> dict[str, float]:
    """Summarize cost resilience and implementation shortfall."""

    aligned = pd.concat(
        [
            pd.to_numeric(gross_returns, errors="coerce").rename("gross"),
            pd.to_numeric(net_returns, errors="coerce").rename("net"),
            pd.to_numeric(turnover, errors="coerce").rename("turnover"),
        ],
        axis=1,
    ).dropna()
    if aligned.empty:
        return _empty_shortfall()
    implied_cost = aligned["gross"] - aligned["net"]
    total_turnover = float(aligned["turnover"].abs().sum())
    total_net = float(aligned["net"].sum())
    total_implied_cost = float(implied_cost.sum())
    if execution_costs is not None:
        explicit_cost = float(
            pd.to_numeric(execution_costs, errors="coerce").reindex(aligned.index).fillna(0.0).sum()
        )
    else:
        explicit_cost = total_implied_cost
    return {
        "total_turnover": total_turnover,
        "implementation_shortfall": total_implied_cost,
        "shortfall_per_turnover": total_implied_cost / total_turnover
        if total_turnover > 0
        else float("nan"),
        "net_return_per_turnover": total_net / total_turnover
        if total_turnover > 0
        else float("nan"),
        "return_on_execution_costs": total_net / explicit_cost
        if explicit_cost > 0
        else float("nan"),
        "cost_break_even_multiple": (total_net + explicit_cost) / explicit_cost
        if explicit_cost > 0
        else float("nan"),
    }


def implied_precision(
    *,
    target_sharpe: float,
    bets_per_year: float,
    average_hit: float,
    average_miss: float,
) -> float:
    """Solve the AFML asymmetric-payout equation for required precision."""

    n = float(bets_per_year)
    theta = float(target_sharpe)
    win = float(average_hit)
    loss = float(average_miss)
    spread = win - loss
    if n <= 0 or spread <= 0:
        return float("nan")
    a = (n + theta**2) * spread**2
    b = (2.0 * n * loss - theta**2 * spread) * spread
    c = n * loss**2
    discriminant = b**2 - 4.0 * a * c
    if discriminant < 0 or a == 0:
        return float("nan")
    roots = [
        (-b + sqrt(discriminant)) / (2.0 * a),
        (-b - sqrt(discriminant)) / (2.0 * a),
    ]
    valid = [root for root in roots if 0.0 <= root <= 1.0]
    return min(valid) if valid else float("nan")


def strategy_failure_probability(
    bet_returns: pd.Series,
    *,
    target_sharpe: float,
    periods_per_year: float,
    evaluation_years: float = 2.0,
    bootstrap_samples: int = 2000,
    random_state: int | np.random.Generator | None = None,
) -> float:
    """Bootstrap the probability that precision falls below the required level."""

    values = pd.to_numeric(bet_returns, errors="coerce").dropna()
    if values.empty or bootstrap_samples <= 0 or evaluation_years <= 0:
        return float("nan")
    hits = values.loc[values > 0]
    misses = values.loc[values <= 0]
    if hits.empty or misses.empty:
        return float("nan")
    years = _elapsed_years(values.index, len(values), periods_per_year)
    bets_per_year = len(values) / years if years > 0 else float(periods_per_year)
    required = implied_precision(
        target_sharpe=target_sharpe,
        bets_per_year=bets_per_year,
        average_hit=float(hits.mean()),
        average_miss=float(misses.mean()),
    )
    if not np.isfinite(required):
        return float("nan")
    sample_size = max(1, round(bets_per_year * evaluation_years))
    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )
    raw = values.to_numpy(dtype=float)
    failures = 0
    for _ in range(bootstrap_samples):
        sample = rng.choice(raw, size=sample_size, replace=True)
        precision = float(np.mean(sample > 0))
        failures += int(precision < required)
    return failures / bootstrap_samples


def summarize_strategy_risk(
    bet_returns: pd.Series,
    *,
    periods_per_year: float = 252.0,
    benchmark_sharpe: float = 0.0,
    target_sharpe: float = 1.0,
    evaluation_years: float = 2.0,
    bootstrap_samples: int = 2000,
    random_state: int | np.random.Generator | None = None,
) -> StrategyRiskReport:
    """Build a compact strategy-risk report from bet-level returns."""

    values = pd.to_numeric(bet_returns, errors="coerce").dropna()
    if values.empty:
        return StrategyRiskReport(
            observations=0,
            periods_per_year=periods_per_year,
            sharpe=float("nan"),
            probabilistic_sharpe=float("nan"),
            positive_return_hhi=float("nan"),
            negative_return_hhi=float("nan"),
            time_concentration_hhi=float("nan"),
            hit_ratio=float("nan"),
            average_hit=float("nan"),
            average_miss=float("nan"),
            bets_per_year=float("nan"),
            strategy_failure_probability=float("nan"),
        )
    std = float(values.std(ddof=1))
    sharpe = float(values.mean() / std * sqrt(periods_per_year)) if std > 0 else float("nan")
    concentration = return_concentration(values)
    hits = values.loc[values > 0]
    misses = values.loc[values <= 0]
    years = _elapsed_years(values.index, len(values), periods_per_year)
    return StrategyRiskReport(
        observations=len(values),
        periods_per_year=periods_per_year,
        sharpe=sharpe,
        probabilistic_sharpe=probabilistic_sharpe_ratio(
            values,
            benchmark_sharpe=benchmark_sharpe,
            periods_per_year=periods_per_year,
        ),
        positive_return_hhi=concentration["positive_return_hhi"],
        negative_return_hhi=concentration["negative_return_hhi"],
        time_concentration_hhi=concentration["time_concentration_hhi"],
        hit_ratio=float((values > 0).mean()),
        average_hit=float(hits.mean()) if not hits.empty else float("nan"),
        average_miss=float(misses.mean()) if not misses.empty else float("nan"),
        bets_per_year=len(values) / years if years > 0 else float("nan"),
        strategy_failure_probability=strategy_failure_probability(
            values,
            target_sharpe=target_sharpe,
            periods_per_year=periods_per_year,
            evaluation_years=evaluation_years,
            bootstrap_samples=bootstrap_samples,
            random_state=random_state,
        ),
    )


def _elapsed_years(index: pd.Index, observations: int, periods_per_year: float) -> float:
    if isinstance(index, pd.DatetimeIndex) and len(index) > 1:
        days = max((index.max() - index.min()).days, 1)
        return days / 365.25
    return observations / periods_per_year


def _empty_shortfall() -> dict[str, float]:
    return {
        "total_turnover": float("nan"),
        "implementation_shortfall": float("nan"),
        "shortfall_per_turnover": float("nan"),
        "net_return_per_turnover": float("nan"),
        "return_on_execution_costs": float("nan"),
        "cost_break_even_multiple": float("nan"),
    }


__all__ = [
    "StrategyRiskReport",
    "implementation_shortfall_metrics",
    "implied_precision",
    "normalized_hhi",
    "probabilistic_sharpe_ratio",
    "return_concentration",
    "strategy_failure_probability",
    "summarize_strategy_risk",
]
