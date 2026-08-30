from pathlib import Path

import pandas as pd
import pytest

from portfolio_backtester import evaluation


def test_portfolio_package_does_not_own_freshness_score_transform() -> None:
    duplicate = Path("src/portfolio_backtester/freshness_overlay.py")
    assert not duplicate.exists(), (
        "freshness score transformation is alpha-owned; portfolio-backtester "
        "must consume an already-transformed score"
    )


def _walk_forward_context(**overrides: object) -> dict[str, object]:
    context: dict[str, object] = {
        "backtest_signal_direction_raw": None,
        "df_full": pd.DataFrame(
            {
                "trade_date": pd.to_datetime(["2026-01-05", "2026-01-06"]),
                "feature": [1.0, 2.0],
            }
        ),
        "features": ["feature"],
        "score_postprocess_method": "none",
        "score_postprocess_columns": [],
        "score_postprocess_strength": 1.0,
        "score_postprocess_min_obs": None,
        "freshness_overlay": {"enabled": True},
    }
    context.update(overrides)
    return context


class _Model:
    def predict(self, frame: pd.DataFrame) -> pd.Series:
        return pd.Series([0.2, 0.8], index=frame.index)


def test_walk_forward_backtest_uses_injected_alpha_freshness_owner() -> None:
    calls: dict[str, object] = {}

    def apply_overlay(
        frame: pd.DataFrame,
        *,
        score_col: str,
        cfg: object,
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        calls.update({"score_col": score_col, "cfg": cfg})
        output = frame.copy()
        output[score_col] = output[score_col] + 10.0
        return output, {"enabled": True}

    scored = evaluation._score_walk_forward_backtest_frame(
        {"test_start": "2026-01-05", "test_end": "2026-01-06"},
        model_w=_Model(),
        direction=1.0,
        context=_walk_forward_context(freshness_overlay_applier=apply_overlay),
    )

    assert scored is not None
    frame, score_col = scored
    assert score_col == "pred"
    assert frame["pred"].tolist() == [10.2, 10.8]
    assert calls == {"score_col": "pred", "cfg": {"enabled": True}}


def test_walk_forward_backtest_fails_closed_without_freshness_owner() -> None:
    with pytest.raises(ValueError, match="freshness_overlay_applier"):
        evaluation._score_walk_forward_backtest_frame(
            {"test_start": "2026-01-05", "test_end": "2026-01-06"},
            model_w=_Model(),
            direction=1.0,
            context=_walk_forward_context(),
        )


def test_evaluation_does_not_import_local_freshness_module() -> None:
    source = Path("src/portfolio_backtester/_evaluation_backtest.py").read_text()

    assert "from .freshness_overlay import" not in source
