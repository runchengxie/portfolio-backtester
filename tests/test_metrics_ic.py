from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_backtester.metrics import (
    leg_attribution_frame,
    summarize_leg_attribution,
)


def _sample_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": ["2024-01-02"] * 6 + ["2024-01-03"] * 6,
            "symbol": list(range(6)) * 2,
            "signal": [10, 8, 6, 4, 2, 0, 10, 8, 6, 4, 2, 0],
            "fwd_ret": [
                0.05,
                0.03,
                0.01,
                -0.01,
                -0.03,
                -0.05,
                0.04,
                0.02,
                0.0,
                -0.02,
                -0.04,
                -0.06,
            ],
        }
    )


def test_leg_attribution_frame_returns_expected_columns() -> None:
    frame = leg_attribution_frame(_sample_data(), "signal", "fwd_ret")

    expected = {
        "top_ret",
        "bottom_ret",
        "cross_mean",
        "spread",
        "top_excess",
        "bottom_drag",
    }
    assert set(frame.columns) == expected
    assert len(frame) == 2
    assert frame.index.name == "trade_date"


def test_leg_attribution_frame_quantile_math() -> None:
    data = _sample_data()
    frame = leg_attribution_frame(data, "signal", "fwd_ret", top_quantile=0.5, bottom_quantile=0.5)
    first = frame.iloc[0]

    assert first["top_ret"] == pytest.approx((0.05 + 0.03 + 0.01) / 3)
    assert first["bottom_ret"] == pytest.approx((-0.01 - 0.03 - 0.05) / 3)
    assert first["cross_mean"] == pytest.approx(0.0)
    assert first["spread"] == pytest.approx(first["top_ret"] - first["bottom_ret"])
    assert first["top_excess"] == pytest.approx(first["top_ret"] - first["cross_mean"])
    assert first["bottom_drag"] == pytest.approx(first["cross_mean"] - first["bottom_ret"])


def test_leg_attribution_frame_empty_input() -> None:
    empty = pd.DataFrame(columns=["trade_date", "signal", "fwd_ret"])
    frame = leg_attribution_frame(empty, "signal", "fwd_ret")

    assert frame.empty


def test_leg_attribution_frame_single_name_per_date() -> None:
    data = pd.DataFrame(
        {
            "trade_date": ["2024-01-02", "2024-01-03"],
            "symbol": [1, 2],
            "signal": [1.0, 2.0],
            "fwd_ret": [0.01, 0.02],
        }
    )
    frame = leg_attribution_frame(data, "signal", "fwd_ret")

    assert frame.empty


def test_summarize_leg_attribution_monthly_aggregation() -> None:
    data = _sample_data()
    frame = leg_attribution_frame(data, "signal", "fwd_ret", top_quantile=0.5, bottom_quantile=0.5)
    summary = summarize_leg_attribution(frame, period="M")

    assert list(summary.columns) == [
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
    assert len(summary) == 1
    assert summary.loc[0, "n_dates"] == 2
    assert summary.loc[0, "bottom_share"] == pytest.approx(
        summary.loc[0, "bottom_drag"] / summary.loc[0, "spread"]
    )


def test_summarize_leg_attribution_empty_input() -> None:
    empty = pd.DataFrame(
        columns=["top_ret", "bottom_ret", "cross_mean", "spread", "top_excess", "bottom_drag"]
    )
    summary = summarize_leg_attribution(empty, period="M")

    assert summary.empty


def test_summarize_leg_attribution_zero_spread_produces_nan_share() -> None:
    frame = pd.DataFrame(
        {
            "top_ret": [0.01, 0.01],
            "bottom_ret": [0.01, 0.01],
            "cross_mean": [0.01, 0.01],
            "spread": [0.0, 0.0],
            "top_excess": [0.0, 0.0],
            "bottom_drag": [0.0, 0.0],
        },
        index=pd.to_datetime(["2024-01-02", "2024-02-02"]),
    )
    summary = summarize_leg_attribution(frame, period="M")

    assert summary["bottom_share"].isna().all()
    assert np.isfinite(summary["n_dates"]).all()
