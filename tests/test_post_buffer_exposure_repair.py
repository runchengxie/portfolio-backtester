from __future__ import annotations

import math

import pandas as pd

from cstree.backtesting.post_buffer_exposure_repair import (
    PostBufferExposureRepairConfig,
    repair_post_buffer_exposure,
)


def _positions(rows: list[dict[str, object]]) -> pd.DataFrame:
    base = {
        "rebalance_date": 20251128,
        "entry_date": 20251201,
        "side": "long",
    }
    return pd.DataFrame([{**base, **row} for row in rows])


def _source(rows: list[dict[str, object]]) -> pd.DataFrame:
    base = {
        "trade_date": 20251128,
        "is_tradable": True,
    }
    return pd.DataFrame([{**base, **row} for row in rows])


def _bank_breach(value: float = -0.205) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "rebalance_date": 20251128,
                "check": "industry_active",
                "name": "银行",
                "value": value,
                "status": "breached",
            }
        ]
    )


def test_bank_repair_uses_strict_guardrail_candidate_before_fallback() -> None:
    positions = _positions(
        [
            {"symbol": "BANK_STRICT", "weight": 0.10, "signal": 0.2, "rank": 2},
            {"symbol": "GROWTH_A", "weight": 0.45, "signal": 1.0, "rank": 1},
            {"symbol": "GROWTH_B", "weight": 0.45, "signal": -0.5, "rank": 3},
        ]
    )
    source = _source(
        [
            {
                "symbol": "BANK_STRICT",
                "first_industry_name": "银行",
                "signal_z": 0.1,
                "earnings_burst_rank": 0.72,
                "exposure_momentum_z": 0.0,
            },
            {
                "symbol": "BANK_FALLBACK",
                "first_industry_name": "银行",
                "signal_z": 2.0,
                "earnings_burst_rank": 0.67,
                "exposure_momentum_z": 0.0,
            },
            {
                "symbol": "GROWTH_A",
                "first_industry_name": "电子",
                "signal_z": 1.0,
                "earnings_burst_rank": 0.80,
                "exposure_momentum_z": 0.0,
            },
            {
                "symbol": "GROWTH_B",
                "first_industry_name": "传媒",
                "signal_z": -0.5,
                "earnings_burst_rank": 0.80,
                "exposure_momentum_z": 0.0,
            },
        ]
    )

    result = repair_post_buffer_exposure(positions, source, _bank_breach())

    action = result.actions[0]
    assert action["status"] == "applied"
    assert action["receiver"] == "BANK_STRICT"
    assert action["fallback_used"] is False
    assert math.isclose(action["moved_weight"], 0.008, rel_tol=0, abs_tol=1e-12)
    bank_weight = result.positions.loc[result.positions["symbol"] == "BANK_STRICT", "weight"].iloc[
        0
    ]
    assert math.isclose(bank_weight, 0.108, rel_tol=0, abs_tol=1e-12)
    assert "BANK_FALLBACK" not in set(result.positions["symbol"])


def test_bank_repair_uses_rank_065_fallback_only_when_strict_pool_is_empty() -> None:
    positions = _positions(
        [
            {"symbol": "GROWTH_A", "weight": 0.50, "signal": 1.0, "rank": 1},
            {"symbol": "GROWTH_B", "weight": 0.50, "signal": -1.0, "rank": 2},
        ]
    )
    source = _source(
        [
            {
                "symbol": "BANK_FALLBACK",
                "first_industry_name": "银行",
                "signal_z": 0.3,
                "earnings_burst_rank": 0.67,
                "exposure_momentum_z": 0.0,
            },
            {
                "symbol": "BANK_WEAK",
                "first_industry_name": "银行",
                "signal_z": 1.0,
                "earnings_burst_rank": 0.64,
                "exposure_momentum_z": 0.0,
            },
            {
                "symbol": "GROWTH_A",
                "first_industry_name": "电子",
                "signal_z": 1.0,
                "earnings_burst_rank": 0.80,
                "exposure_momentum_z": 0.0,
            },
            {
                "symbol": "GROWTH_B",
                "first_industry_name": "传媒",
                "signal_z": -1.0,
                "earnings_burst_rank": 0.80,
                "exposure_momentum_z": 0.0,
            },
        ]
    )

    result = repair_post_buffer_exposure(positions, source, _bank_breach(value=-0.201))

    action = result.actions[0]
    assert action["status"] == "applied"
    assert action["receiver"] == "BANK_FALLBACK"
    assert action["fallback_used"] is True
    assert action["fallback_min_rank"] == 0.65
    assert math.isclose(action["moved_weight"], 0.004, rel_tol=0, abs_tol=1e-12)
    assert "BANK_WEAK" not in set(result.positions["symbol"])
    assert math.isclose(result.positions["weight"].sum(), 1.0, rel_tol=0, abs_tol=1e-12)


def test_bank_repair_leaves_breach_when_no_strict_or_fallback_candidate_exists() -> None:
    positions = _positions(
        [
            {"symbol": "GROWTH_A", "weight": 0.50, "signal": 1.0, "rank": 1},
            {"symbol": "GROWTH_B", "weight": 0.50, "signal": -1.0, "rank": 2},
        ]
    )
    source = _source(
        [
            {
                "symbol": "BANK_NEGATIVE_SIGNAL",
                "first_industry_name": "银行",
                "signal_z": -0.1,
                "earnings_burst_rank": 0.68,
                "exposure_momentum_z": 0.0,
            },
            {
                "symbol": "GROWTH_A",
                "first_industry_name": "电子",
                "signal_z": 1.0,
                "earnings_burst_rank": 0.80,
                "exposure_momentum_z": 0.0,
            },
            {
                "symbol": "GROWTH_B",
                "first_industry_name": "传媒",
                "signal_z": -1.0,
                "earnings_burst_rank": 0.80,
                "exposure_momentum_z": 0.0,
            },
        ]
    )

    result = repair_post_buffer_exposure(positions, source, _bank_breach(value=-0.201))

    assert result.actions[0]["status"] == "no_candidate"
    assert result.actions[0]["fallback_used"] is True
    assert set(result.positions["symbol"]) == {"GROWTH_A", "GROWTH_B"}


def test_momentum_repair_moves_minimum_weight_plus_margin() -> None:
    positions = _positions(
        [
            {"symbol": "LOW_MOM", "weight": 0.10, "signal": -1.0, "rank": 3},
            {"symbol": "MID_A", "weight": 0.45, "signal": 1.0, "rank": 1},
            {"symbol": "MID_B", "weight": 0.45, "signal": 0.5, "rank": 2},
        ]
    )
    source = _source(
        [
            {
                "symbol": "LOW_MOM",
                "first_industry_name": "传媒",
                "signal_z": -1.0,
                "earnings_burst_rank": 0.80,
                "exposure_momentum_z": -2.0,
            },
            {
                "symbol": "MID_A",
                "first_industry_name": "电子",
                "signal_z": 1.0,
                "earnings_burst_rank": 0.80,
                "exposure_momentum_z": 0.0,
            },
            {
                "symbol": "MID_B",
                "first_industry_name": "机械",
                "signal_z": 0.5,
                "earnings_burst_rank": 0.80,
                "exposure_momentum_z": 0.0,
            },
            {
                "symbol": "HIGH_MOM",
                "first_industry_name": "电子",
                "signal_z": 0.7,
                "earnings_burst_rank": 0.75,
                "exposure_momentum_z": 2.0,
            },
        ]
    )
    breaches = pd.DataFrame(
        [
            {
                "rebalance_date": 20251128,
                "check": "style_active",
                "name": "momentum",
                "value": -1.10,
                "status": "breached",
            }
        ]
    )

    result = repair_post_buffer_exposure(positions, source, breaches)

    action = result.actions[0]
    expected_move = 0.103 / 4.0
    assert action["status"] == "applied"
    assert math.isclose(action["moved_weight"], expected_move, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(
        action["achieved_exposure_delta_approx"],
        0.103,
        rel_tol=0,
        abs_tol=1e-12,
    )
    high_weight = result.positions.loc[result.positions["symbol"] == "HIGH_MOM", "weight"].iloc[0]
    low_weight = result.positions.loc[result.positions["symbol"] == "LOW_MOM", "weight"].iloc[0]
    assert math.isclose(high_weight, expected_move, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(low_weight, 0.10 - expected_move, rel_tol=0, abs_tol=1e-12)


def test_momentum_repair_derives_missing_guardrail_from_signal_backtest() -> None:
    positions = _positions(
        [
            {"symbol": "LOW_MOM", "weight": 0.10, "signal": -1.0, "rank": 3},
            {"symbol": "MID_A", "weight": 0.45, "signal": 1.0, "rank": 1},
            {"symbol": "MID_B", "weight": 0.45, "signal": 0.5, "rank": 2},
        ]
    )
    source = _source(
        [
            {
                "symbol": "LOW_MOM",
                "first_industry_name": "传媒",
                "signal_backtest": -1.0,
                "exposure_momentum_z": -2.0,
            },
            {
                "symbol": "MID_A",
                "first_industry_name": "电子",
                "signal_backtest": 1.0,
                "exposure_momentum_z": 0.0,
            },
            {
                "symbol": "MID_B",
                "first_industry_name": "机械",
                "signal_backtest": 0.5,
                "exposure_momentum_z": 0.0,
            },
            {
                "symbol": "HIGH_MOM",
                "first_industry_name": "电子",
                "signal_backtest": 0.7,
                "exposure_momentum_z": 2.0,
            },
        ]
    )
    breaches = pd.DataFrame(
        [
            {
                "rebalance_date": 20251128,
                "check": "style_active",
                "name": "momentum",
                "value": -1.10,
                "status": "breached",
            }
        ]
    )

    result = repair_post_buffer_exposure(
        positions,
        source,
        breaches,
        config=PostBufferExposureRepairConfig(signal_col="signal_backtest"),
    )

    action = result.actions[0]
    assert action["status"] == "applied"
    assert "HIGH_MOM" in set(result.positions["symbol"])
