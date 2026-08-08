from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_backtester.portfolio import build_position_weights


def test_signal_weighting_concentrates_on_high_score_names():
    symbols = [f"S{i:03d}" for i in range(100)]
    day = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-01"] * 100),
            "symbol": symbols,
            "score": list(range(100, 0, -1)),
            "medadv20_amount": np.geomspace(100.0, 10_000.0, 100),
        }
    )

    weights = build_position_weights(
        day,
        symbols,
        "score",
        side="long",
        weighting="signal",
    )

    assert weights.sum() == pytest.approx(1.0)
    # high-score names get more weight (signal, not equal)
    assert float(weights.iloc[0]) > float(weights.iloc[-1])
    # not equal-weight
    assert weights.iloc[0] != pytest.approx(1.0 / 100)
    # top-10 avg weight exceeds bottom-10
    assert weights.iloc[:10].mean() > weights.iloc[-10:].mean()


def test_signal_weighting_sums_to_one_for_short_side():
    symbols = [f"S{i:03d}" for i in range(50)]
    day = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-01"] * 50),
            "symbol": symbols,
            "score": list(range(50, 0, -1)),
        }
    )

    weights = build_position_weights(
        day,
        symbols,
        "score",
        side="short",
        weighting="signal",
    )

    assert weights.sum() == pytest.approx(1.0)
    # short 侧:原始高分(信号为正)在 short 侧反转,低分权重大
    assert float(weights.iloc[-1]) > float(weights.iloc[0])


def test_signal_weighting_handles_nan_scores():
    symbols = [f"S{i:03d}" for i in range(10)]
    scores = [10.0, 9.0, np.nan, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    day = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-01"] * 10),
            "symbol": symbols,
            "score": scores,
        }
    )

    weights = build_position_weights(
        day,
        symbols,
        "score",
        side="long",
        weighting="signal",
    )

    assert weights.sum() == pytest.approx(1.0)
    # NaN 分数被填充,不导致崩溃
    assert weights.isna().sum() == 0
