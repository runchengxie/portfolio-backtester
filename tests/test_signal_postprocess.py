import numpy as np
import pandas as pd
import pytest

from portfolio_backtester.signal_postprocess import (
    apply_score_postprocess,
    neutralize_cross_sectional_series,
    rank_blend_cross_sectional_series,
)


def test_neutralize_cross_sectional_series_removes_linear_size_component():
    df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2020-01-01"] * 4 + ["2020-01-02"] * 4),
            "pred": [1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0],
            "log_mcap": [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
        }
    )

    neutralized = neutralize_cross_sectional_series(
        df,
        "pred",
        ["log_mcap"],
        strength=1.0,
        min_obs=4,
    )
    out = df.assign(pred_adj=neutralized)

    for _, group in out.groupby("trade_date", sort=False):
        assert float(group["pred_adj"].std(ddof=0)) < 1e-8


def test_apply_score_postprocess_strength_zero_returns_original_series():
    df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2020-01-01"] * 3),
            "pred": [1.0, 2.0, 3.0],
            "log_mcap": [10.0, 11.0, 12.0],
        }
    )

    out = apply_score_postprocess(
        df,
        "pred",
        method="neutralize",
        columns=["log_mcap"],
        strength=0.0,
        min_obs=3,
    )

    assert out.tolist() == df["pred"].tolist()


def test_rank_blend_cross_sectional_series_blends_base_and_overlay_ranks_by_date():
    df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2020-01-01"] * 3 + ["2020-01-08"] * 3),
            "pred": [1.0, 2.0, 3.0, 3.0, 1.0, 2.0],
            "volume_heat": [30.0, 10.0, 20.0, 10.0, 30.0, 20.0],
            "industry_heat": [3.0, 2.0, 1.0, np.nan, np.nan, np.nan],
        }
    )

    out = rank_blend_cross_sectional_series(
        df,
        "pred",
        ["volume_heat", "industry_heat"],
        strength=0.05,
    )

    base_rank = df.groupby("trade_date")["pred"].rank(method="average", pct=True)
    overlay_ranks = df.groupby("trade_date")[["volume_heat", "industry_heat"]].rank(
        method="average",
        pct=True,
    )
    expected = 0.95 * base_rank + 0.05 * overlay_ranks.mean(axis=1, skipna=True)

    assert out.tolist() == pytest.approx(expected.tolist())


def test_apply_score_postprocess_rank_blend_falls_back_to_base_rank_when_overlay_is_missing():
    df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2020-01-01"] * 3),
            "pred": [1.0, 2.0, 3.0],
            "volume_heat": [np.nan, np.nan, np.nan],
        }
    )

    out = apply_score_postprocess(
        df,
        "pred",
        method="rank_blend",
        columns=["volume_heat"],
        strength=0.05,
    )

    expected = df.groupby("trade_date")["pred"].rank(method="average", pct=True)
    assert out.tolist() == pytest.approx(expected.tolist())
