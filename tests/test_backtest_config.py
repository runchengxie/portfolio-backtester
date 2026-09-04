from __future__ import annotations

import pytest

from portfolio_backtester.backtest_config import resolve_backtest_base_settings


def test_resolve_backtest_base_settings_defaults_and_overrides() -> None:
    settings = resolve_backtest_base_settings(
        {
            "benchmark_symbol": " 000300.SH ",
            "weighting": "signal",
            "tearsheet": {"enabled": True},
            "benchmark_compare": [{"symbol": "000905.SH"}],
            "post_buffer_exposure_repair": True,
        },
        eval_top_k=20,
        eval_rebalance_frequency="W",
        eval_transaction_cost_bps=10.0,
        label_horizon_days=5,
    )

    assert settings["BACKTEST_BENCHMARK"] == "000300.SH"
    assert settings["BACKTEST_WEIGHTING"] == "signal"
    assert settings["BACKTEST_TEARSHEET_ENABLED"] is True
    assert settings["BACKTEST_BENCHMARK_COMPARE"] == [
        {"name": "000905.SH", "source_type": "symbol", "symbol": "000905.SH"}
    ]
    assert settings["BACKTEST_POST_BUFFER_EXPOSURE_REPAIR"] == {"enabled": True}


def test_resolve_backtest_base_settings_rejects_invalid_options() -> None:
    with pytest.raises(SystemExit, match="weighting"):
        resolve_backtest_base_settings(
            {"weighting": "random"},
            eval_top_k=20,
            eval_rebalance_frequency="W",
            eval_transaction_cost_bps=10.0,
            label_horizon_days=5,
        )

    with pytest.raises(SystemExit, match="mutually exclusive"):
        resolve_backtest_base_settings(
            {"benchmark_symbol": "000300.SH", "benchmark_returns_file": "benchmark.csv"},
            eval_top_k=20,
            eval_rebalance_frequency="W",
            eval_transaction_cost_bps=10.0,
            label_horizon_days=5,
        )
