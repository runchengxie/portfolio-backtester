from __future__ import annotations

import pytest

from portfolio_backtester.execution_config import resolve_execution_settings


def _settings() -> dict[str, object]:
    return {
        "BACKTEST_COST_BPS": 10.0,
        "BACKTEST_EXIT_PRICE_POLICY": "strict",
        "BACKTEST_EXIT_FALLBACK_POLICY": "ffill",
        "BACKTEST_TRADABLE_COL": "is_tradable",
    }


def test_resolve_execution_settings_merges_backtest_overrides() -> None:
    resolved = resolve_execution_settings(
        execution_cfg={"cost": {"type": "bps", "bps": 5}},
        backtest_cfg={"execution": {"cost": {"bps": 7}}},
        backtest_settings=_settings(),
        price_col="close",
    )

    assert resolved["BACKTEST_EXECUTION_SOURCE"] == "explicit_execution_config"
    assert resolved["BACKTEST_COST_BPS_EFFECTIVE"] == pytest.approx(7.0)
    assert "close" in resolved["EXECUTION_PRICING_COLS"]


def test_resolve_execution_settings_rejects_invalid_exit_policy() -> None:
    settings = _settings()
    settings["BACKTEST_EXIT_PRICE_POLICY"] = "unknown"
    with pytest.raises(SystemExit, match="exit_price_policy"):
        resolve_execution_settings(
            execution_cfg={},
            backtest_cfg={},
            backtest_settings=settings,
            price_col="close",
        )
