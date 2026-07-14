from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio_backtester.bet_sizing import (
    SizingConfig,
    average_active_bets,
    build_sized_weights,
    discretize_weights,
)
from portfolio_backtester.evidence_receipts import build_portfolio_sizing_receipt
from portfolio_backtester.hrp import HrpConfig, hierarchical_risk_parity, rolling_hrp_weights
from portfolio_backtester.portfolio_weights import build_position_weights
from portfolio_backtester.strategy_risk import (
    implementation_shortfall_metrics,
    probabilistic_sharpe_ratio,
    return_concentration,
    summarize_strategy_risk,
)


def test_probability_volatility_sizing_applies_caps_and_discretization() -> None:
    frame = pd.DataFrame(
        {
            "signal": [0.3, 0.2, 0.1],
            "calibrated_probability": [0.8, 0.7, 0.55],
            "predicted_volatility": [0.2, 0.1, 0.15],
        },
        index=["A", "B", "C"],
    )
    weights = build_sized_weights(
        frame,
        score_col="signal",
        config=SizingConfig(
            method="probability_vol_target",
            single_name_cap=0.5,
            step_size=0.01,
        ),
    )
    assert np.isclose(weights.sum(), 1.0)
    assert weights.max() <= 0.5 + 1e-12
    assert np.allclose((weights / 0.01).round(), weights / 0.01, atol=1e-8)


def test_calibrated_weighting_mode_is_used_by_position_builder() -> None:
    day = pd.DataFrame(
        {
            "symbol": ["A", "B", "C"],
            "signal": [0.3, 0.2, 0.1],
            "calibrated_probability": [0.8, 0.7, 0.55],
            "predicted_volatility": [0.2, 0.1, 0.15],
        }
    )
    weights = build_position_weights(
        day,
        ["A", "B", "C"],
        "signal",
        side="long",
        weighting="probability_vol_target",
    )

    assert np.isclose(weights.sum(), 1.0)
    assert weights.index.tolist() == ["A", "B", "C"]
    assert weights["B"] > weights["A"] > weights["C"]


def test_sizing_receipt_supports_legacy_and_calibrated_methods() -> None:
    weights = pd.Series([0.5, 0.3, 0.2], index=["A", "B", "C"])
    receipt = build_portfolio_sizing_receipt(
        weights,
        method="equal",
        configuration={"top_k": 3},
    )

    assert receipt["method"] == "equal"
    assert receipt["target_count"] == 3
    assert receipt["gross_exposure"] == 1.0
    assert receipt["weights_sha256"]


def test_active_bets_are_averaged_and_discretized() -> None:
    events = pd.DataFrame(
        {
            "label_start": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "label_end": pd.to_datetime(["2024-01-03", "2024-01-04"]),
            "bet_size": [1.0, -0.5],
        }
    )
    active = average_active_bets(events, time_index=pd.date_range("2024-01-01", "2024-01-04"))
    assert active.loc["2024-01-01"] == 1.0
    assert active.loc["2024-01-02"] == 0.25
    assert discretize_weights(active, step_size=0.25).loc["2024-01-02"] == 0.25


def test_hrp_weights_are_point_in_time_and_sum_to_one() -> None:
    rng = np.random.default_rng(7)
    index = pd.date_range("2023-01-01", periods=300)
    returns = pd.DataFrame(
        {
            "value": rng.normal(0, 0.01, len(index)),
            "quality": rng.normal(0, 0.008, len(index)),
            "momentum": rng.normal(0, 0.012, len(index)),
        },
        index=index,
    )
    result = hierarchical_risk_parity(
        returns,
        config=HrpConfig(shrinkage=0.1, max_weight=0.6),
    )
    assert np.isclose(result.weights.sum(), 1.0)
    assert set(result.weights.index) == set(returns.columns)

    rolling = rolling_hrp_weights(
        returns,
        pd.DatetimeIndex([index[150], index[250]]),
        lookback=120,
        min_observations=60,
    )
    assert len(rolling) == 2
    assert np.allclose(rolling.sum(axis=1), 1.0)


def test_strategy_risk_report_and_shortfall_metrics() -> None:
    index = pd.date_range("2023-01-01", periods=100)
    returns = pd.Series(np.tile([0.01, 0.008, -0.006, 0.004], 25), index=index)
    psr = probabilistic_sharpe_ratio(returns)
    concentration = return_concentration(returns)
    report = summarize_strategy_risk(
        returns,
        periods_per_year=252,
        bootstrap_samples=100,
        random_state=1,
    )
    assert 0.0 <= psr <= 1.0
    assert report.hit_ratio == 0.75
    assert concentration["positive_return_hhi"] >= 0.0

    shortfall = implementation_shortfall_metrics(
        gross_returns=returns + 0.001,
        net_returns=returns,
        turnover=pd.Series(0.2, index=index),
    )
    assert shortfall["implementation_shortfall"] > 0
    assert shortfall["shortfall_per_turnover"] > 0
