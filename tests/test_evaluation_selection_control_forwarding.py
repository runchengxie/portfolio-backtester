from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from portfolio_backtester import evaluation


def _evaluation_context() -> dict[str, Any]:
    return {
        "backtest_enabled": True,
        "live_enabled": False,
        "backtest_rebalance_frequency": "D",
        "backtest_pricing_df": pd.DataFrame({"close": [100.0]}),
        "backtest_tradable_col": "is_tradable",
        "backtest_group_col": "industry",
        "backtest_max_names_per_group": None,
        "backtest_top_k": 10,
        "backtest_cost_bps_effective": 0.0,
        "backtest_trading_days_per_year": 252,
        "backtest_weighting": "equal",
        "backtest_exit_mode": "rebalance",
        "backtest_exit_horizon_days": None,
        "backtest_long_only": True,
        "backtest_short_k": None,
        "backtest_buffer_exit": None,
        "backtest_buffer_entry": None,
        "backtest_exit_price_policy": "strict",
        "backtest_exit_fallback_policy": "ffill",
        "execution_model": object(),
        "price_col": "close",
        "label_shift_days": 0,
    }


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, (None, "legacy_concentrate", None)),
        (
            {
                "backtest_selection_score_margin_col": "candidate_relevance",
                "backtest_max_new_names_shortfall_policy": "carry",
                "backtest_max_positive_names": 10,
            },
            ("candidate_relevance", "carry", 10),
        ),
    ],
)
def test_evaluation_adapters_forward_selection_controls(
    overrides: dict[str, Any],
    expected: tuple[str | None, str, int | None],
) -> None:
    context = _evaluation_context() | overrides
    frame = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2026-07-17"]),
            "symbol": ["000001.SZ"],
            "close": [10.0],
            "signal_backtest": [1.0],
        }
    )
    captured: list[dict[str, Any]] = []

    def capture_backtest(*_args: Any, **kwargs: Any) -> object:
        captured.append(kwargs)
        return object()

    def capture_positions(*_args: Any, **kwargs: Any) -> pd.DataFrame:
        captured.append(kwargs)
        return pd.DataFrame()

    evaluation._run_walk_forward_backtest_topk(
        frame,
        bt_pred_col="signal_backtest",
        context=context,
        valid_dates_set=set(),
        backtest_topk_fn=capture_backtest,
    )
    evaluation._build_period_positions(
        eval_df_full=frame,
        bt_rebalance=[pd.Timestamp("2026-07-17")],
        context=context,
        allow_live_fallback=False,
        build_positions_by_rebalance_fn=capture_positions,
    )
    period_context = context | {"backtest_topk_fn": capture_backtest}
    evaluation._run_period_backtest(
        eval_df_full=frame,
        bt_rebalance=[pd.Timestamp("2026-07-17")],
        context=period_context,
        label_prefix="",
    )

    assert len(captured) == 3
    for kwargs in captured:
        actual = (
            kwargs["selection_score_margin_col"],
            kwargs["max_new_names_shortfall_policy"],
            kwargs["max_positive_names"],
        )
        assert actual == expected
