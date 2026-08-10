"""Phase 5: capacity calibration and concentration grouping tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from portfolio_backtester._capacity_report_config import (
    _alpha_retention_capacity,
    _break_even_capacity,
    _fill_rate_capacity,
    _marginal_impact,
)
from portfolio_backtester._capacity_report_grid import build_capacity_report


def _write_run(
    tmp_path: Path,
    *,
    with_industry: bool = False,
    portfolio_values: list[float] | None = None,
) -> tuple[Path, Path, Path]:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    symbols = ["AAA", "BBB", "CCC"]
    industry = ["tech", "tech", "fin"]
    positions = pd.DataFrame(
        {
            "rebalance_date": ["20200101"] * 3,
            "entry_date": ["20200102"] * 3,
            "symbol": symbols,
            "weight": [0.4, 0.4, 0.2],
            "side": ["long"] * 3,
            **({"industry": industry} if with_industry else {}),
        }
    )
    positions_path = run_dir / "positions_by_rebalance.csv"
    positions.to_csv(positions_path, index=False)

    dates = pd.date_range("2020-01-01", periods=10, freq="B")
    rows = []
    for date in dates:
        for symbol in symbols:
            rows.append(
                {
                    "trade_date": date.strftime("%Y%m%d"),
                    "symbol": symbol,
                    "close": 10.0,
                    "amount": 100_000_000.0,
                    "medadv20_amount": 100_000_000.0,
                    "is_tradable": True,
                    "is_buy_tradable": True,
                    "is_sell_tradable": True,
                }
            )
    pricing_path = run_dir / "pricing.csv"
    pd.DataFrame(rows).to_csv(pricing_path, index=False)

    config = {
        "market": "a_share",
        "data": {"price_col": "close", **({"industry_col": "industry"} if with_industry else {})},
        "backtest": {
            "enabled": True,
            "transaction_cost_bps": 0,
            "trading_days_per_year": 252,
            "tradable_col": "is_tradable",
            "execution_sim": {
                "enabled": True,
                "portfolio_values": portfolio_values or [1_000_000.0, 10_000_000.0, 100_000_000.0],
                "participation_rate": 0.05,
                "liquidity_cols": ["medadv20_amount"],
                "buy_max_days": 2,
                "sell_max_days": 2,
                "zero_fill_abort_days_buy": 2,
            },
        },
    }
    config_path = run_dir / "config.used.yml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (run_dir / "summary.json").write_text(
        json.dumps({"positions": {"by_rebalance_file": str(positions_path)}}),
        encoding="utf-8",
    )
    return run_dir, positions_path, pricing_path


def _fake_rows() -> list[dict]:
    # Synthetic grid rows at primary participation rate (0.05), ascending portfolio_value.
    return [
        {
            "participation_rate": 0.05,
            "portfolio_value": 1.0e6,
            "exec_total_return": 0.10,
            "fill_ratio": 0.99,
            "return_retention": 0.95,
            "sharpe_retention": 0.92,
        },
        {
            "participation_rate": 0.05,
            "portfolio_value": 1.0e7,
            "exec_total_return": 0.05,
            "fill_ratio": 0.97,
            "return_retention": 0.88,
            "sharpe_retention": 0.85,
        },
        {
            "participation_rate": 0.05,
            "portfolio_value": 1.0e8,
            "exec_total_return": -0.02,
            "fill_ratio": 0.80,
            "return_retention": 0.70,
            "sharpe_retention": 0.60,
        },
    ]


def test_break_even_capacity_interpolates_between_grid_points() -> None:
    rows = _fake_rows()
    cap = _break_even_capacity(rows, primary_participation_rate=0.05)
    # Break-even sits between 1e7 (return 0.05) and 1e8 (return -0.02): interpolate.
    assert cap is not None
    assert 10_000_000.0 <= cap <= 100_000_000.0


def test_fill_rate_capacity_respects_target() -> None:
    rows = _fake_rows()
    cap = _fill_rate_capacity(rows, primary_participation_rate=0.05, target=0.95)
    # 1e6 (0.99) and 1e7 (0.97) satisfy; 1e8 (0.80) fails. Interpolation refines
    # between 1e7 and 1e8, so the estimate lies strictly above 1e7 and below 1e8.
    assert cap is not None
    assert 10_000_000.0 < cap < 100_000_000.0


def test_alpha_retention_capacity_uses_configured_metric() -> None:
    rows = _fake_rows()
    cap = _alpha_retention_capacity(
        rows, primary_participation_rate=0.05, target=0.90, metric_key="return_retention"
    )
    # Only 1e6 (0.95) satisfies >= 0.90; 1e7 (0.88) and 1e8 (0.70) fail. Interpolated in (1e6, 1e7).
    assert cap is not None
    assert 1_000_000.0 < cap < 10_000_000.0
    sharpe_cap = _alpha_retention_capacity(
        rows, primary_participation_rate=0.05, target=0.90, metric_key="sharpe_retention"
    )
    # Only 1e6 (0.92) satisfies >= 0.90; interpolated in (1e6, 1e7).
    assert sharpe_cap is not None
    assert 1_000_000.0 < sharpe_cap < 10_000_000.0


def test_marginal_impact_negative_when_capacity_stressed() -> None:
    rows = _fake_rows()
    impact = _marginal_impact(rows, primary_participation_rate=0.05)
    assert impact["marginal_return_per_unit_capital"] is not None
    # Return drops from 0.05 to -0.02 across 90m of capital => negative slope.
    assert impact["marginal_return_per_unit_capital"] < 0.0


def test_concentration_by_symbol_always_present(tmp_path: Path) -> None:
    run_dir, positions_path, pricing_path = _write_run(tmp_path, with_industry=False)
    payload = build_capacity_report(
        run_dir=run_dir,
        config_path=run_dir / "config.used.yml",
        positions_path=positions_path,
        pricing_path=pricing_path,
        portfolio_values=[1_000_000.0, 10_000_000.0, 100_000_000.0],
        participation_rates=[0.05],
        liquidity_cols=["medadv20_amount"],
        threshold_profile="neutral",
        primary_participation_rate=0.05,
        output_csv=None,
    )
    conc = payload["concentration"]
    assert conc["by_symbol"], "symbol concentration must be present"
    assert conc["by_liquidity"] is not None, "liquidity grouping when liquidity col exists"
    assert conc["by_industry"] is None, "industry grouping absent without industry col"
    shares = sum(item["share"] for item in conc["by_symbol"])
    assert abs(shares - 1.0) < 1e-6


def test_concentration_includes_industry_when_column_present(tmp_path: Path) -> None:
    run_dir, positions_path, pricing_path = _write_run(tmp_path, with_industry=True)
    payload = build_capacity_report(
        run_dir=run_dir,
        config_path=run_dir / "config.used.yml",
        positions_path=positions_path,
        pricing_path=pricing_path,
        portfolio_values=[1_000_000.0, 10_000_000.0, 100_000_000.0],
        participation_rates=[0.05],
        liquidity_cols=["medadv20_amount"],
        threshold_profile="neutral",
        primary_participation_rate=0.05,
        output_csv=None,
        industry_col="industry",
    )
    conc = payload["concentration"]
    assert conc["by_industry"] is not None, "industry grouping produced when column supplied"
    assert conc["by_industry"]["hhi"] is not None


def test_capacity_calibration_present_in_payload(tmp_path: Path) -> None:
    run_dir, positions_path, pricing_path = _write_run(tmp_path)
    payload = build_capacity_report(
        run_dir=run_dir,
        config_path=run_dir / "config.used.yml",
        positions_path=positions_path,
        pricing_path=pricing_path,
        portfolio_values=[1_000_000.0, 10_000_000.0, 100_000_000.0],
        participation_rates=[0.05],
        liquidity_cols=["medadv20_amount"],
        threshold_profile="neutral",
        primary_participation_rate=0.05,
        output_csv=None,
    )
    calib = payload["capacity_calibration"]
    for key in (
        "break_even_capacity",
        "fill_rate_95_capacity",
        "alpha_retention_90_capacity",
        "sharpe_retention_90_capacity",
        "marginal_return_per_unit_capital",
        "marginal_sharpe_retention_per_unit_capital",
    ):
        assert key in calib
