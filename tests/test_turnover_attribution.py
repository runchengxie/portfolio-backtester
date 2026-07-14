from __future__ import annotations

import pandas as pd
import pytest

from portfolio_backtester.turnover_attribution import compute_turnover_attribution


def test_compute_turnover_attribution_explains_window_industry_feature_and_regime() -> None:
    positions = pd.DataFrame(
        {
            "rebalance_date": [
                "20200101",
                "20200101",
                "20200108",
                "20200108",
                "20200115",
                "20200115",
            ],
            "entry_date": [
                "20200102",
                "20200102",
                "20200109",
                "20200109",
                "20200116",
                "20200116",
            ],
            "symbol": ["A", "B", "B", "C", "A", "C"],
            "weight": [0.5, 0.5, 0.6, 0.4, 0.3, 0.7],
            "rank": [1, 2, 1, 3, 2, 1],
        }
    )
    scored = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(
                ["20200101", "20200101", "20200108", "20200108", "20200115", "20200115"]
            ),
            "symbol": ["A", "B", "B", "C", "A", "C"],
            "first_industry_name": ["Tech", "Bank", "Bank", "Tech", "Tech", "Tech"],
            "momentum_20d": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    result = compute_turnover_attribution(
        positions,
        scored,
        feature_importance=pd.DataFrame({"feature": ["momentum_20d"], "importance": [1.0]}),
    )

    assert result.summary["status"] == "ok"
    assert result.summary["windows"] == 2
    assert result.by_window["turnover"].tolist() == pytest.approx([1.0, 1.2])
    assert set(result.by_industry["industry"]) == {"Bank", "Tech"}
    assert result.by_feature["feature"].unique().tolist() == ["momentum_20d"]
    assert set(result.by_regime["turnover_regime"]) == {"high_turnover", "low_turnover"}
