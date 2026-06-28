import pandas as pd
import pytest

from cstree.backtesting.freshness_overlay import apply_freshness_overlay


def test_volume_only_freshness_overlay_blends_base_and_volume_ranks():
    frame = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-01"] * 3),
            "symbol": ["A", "B", "C"],
            "signal_backtest": [3.0, 2.0, 1.0],
            "volume_sma5_ratio": [1.0, 3.0, 2.0],
            "volume_sma20_ratio": [1.0, 3.0, 2.0],
            "volume_sma60_ratio": [1.0, 3.0, 2.0],
        }
    )

    overlaid, meta = apply_freshness_overlay(
        frame,
        score_col="signal_backtest",
        cfg={
            "enabled": True,
            "name": "volume_only_lambda_0p05",
            "lambda": 0.05,
            "volume_rank_cols": [
                "volume_sma5_ratio",
                "volume_sma20_ratio",
                "volume_sma60_ratio",
            ],
        },
    )

    assert meta["enabled"] is True
    assert meta["name"] == "volume_only_lambda_0p05"
    assert overlaid["signal_backtest_base"].tolist() == [3.0, 2.0, 1.0]
    assert overlaid["signal_backtest"].tolist() == pytest.approx(
        [
            0.95 * 1.0 + 0.05 * (1 / 3),
            0.95 * (2 / 3) + 0.05 * 1.0,
            0.95 * (1 / 3) + 0.05 * (2 / 3),
        ]
    )


def test_freshness_overlay_fails_when_enabled_columns_are_missing():
    frame = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-01"]),
            "signal_backtest": [1.0],
        }
    )

    with pytest.raises(ValueError, match="missing volume columns"):
        apply_freshness_overlay(
            frame,
            score_col="signal_backtest",
            cfg={"enabled": True, "volume_rank_cols": ["volume_sma5_ratio"]},
        )
