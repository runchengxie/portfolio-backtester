from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from portfolio_backtester import capacity_report


def _write_capacity_run(tmp_path: Path) -> tuple[Path, Path]:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    positions = pd.DataFrame(
        {
            "rebalance_date": ["20200101", "20200101"],
            "entry_date": ["20200102", "20200102"],
            "symbol": ["AAA", "BBB"],
            "weight": [0.5, 0.5],
            "side": ["long", "long"],
        }
    )
    positions_path = run_dir / "positions_by_rebalance.csv"
    positions.to_csv(positions_path, index=False)

    dates = pd.date_range("2020-01-01", periods=8, freq="B")
    rows = []
    for idx, date in enumerate(dates):
        for symbol in ("AAA", "BBB"):
            rows.append(
                {
                    "trade_date": date.strftime("%Y%m%d"),
                    "symbol": symbol,
                    "close": 10.0 + idx * 0.2,
                    "amount": 100_000_000.0,
                    "medadv20_amount": 100_000_000.0,
                    "is_tradable": True,
                    "is_buy_tradable": True,
                    "is_sell_tradable": True,
                }
            )
    pricing_path = tmp_path / "pricing.csv"
    pd.DataFrame(rows).to_csv(pricing_path, index=False)

    config = {
        "market": "a_share",
        "data": {"price_col": "close"},
        "backtest": {
            "enabled": True,
            "transaction_cost_bps": 0,
            "trading_days_per_year": 252,
            "tradable_col": "is_tradable",
            "execution_sim": {
                "enabled": True,
                "portfolio_values": [1_000_000, 100_000_000],
                "participation_rate": 0.05,
                "liquidity_cols": ["medadv20_amount"],
                "cap_daily_amount": False,
                "buy_max_days": 2,
                "sell_max_days": 2,
                "zero_fill_abort_days_buy": 2,
            },
        },
    }
    (run_dir / "config.used.yml").write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps({"positions": {"by_rebalance_file": str(positions_path)}}),
        encoding="utf-8",
    )
    return run_dir, pricing_path


def test_capacity_report_consumes_portfolio_values_and_writes_outputs(tmp_path: Path) -> None:
    run_dir, pricing_path = _write_capacity_run(tmp_path)

    payload = capacity_report.run(
        argparse.Namespace(
            run_dir=str(run_dir),
            config=None,
            positions_file=None,
            pricing_file=str(pricing_path),
            portfolio_value=None,
            participation_rate=None,
            liquidity_col=None,
            primary_participation_rate=None,
            threshold_profile="neutral",
            output_dir=None,
            output_csv=None,
            output_json=None,
            market=None,
            log_level="INFO",
        )
    )

    assert payload["schema"] == "a_share.capacity.v1"
    assert payload["status"] == "passed"
    assert payload["portfolio_grid"] == [1_000_000.0, 100_000_000.0]
    assert payload["participation_rate_grid"] == [0.05]
    assert payload["recommended_capacity"] == 1_000_000.0
    assert payload["hard_capacity"] == 100_000_000.0
    assert "fill_ratio" in payload["binding_constraints"]
    assert (run_dir / "capacity_grid.csv").exists()
    assert (run_dir / "capacity_report.json").exists()

    grid = pd.read_csv(run_dir / "capacity_grid.csv")
    assert grid["portfolio_value"].tolist() == [1_000_000.0, 100_000_000.0]
    assert grid["passed"].tolist() == [True, False]


def test_capacity_report_treats_target_cash_as_intended_overlay(tmp_path: Path) -> None:
    run_dir = tmp_path / "cash_overlay_run"
    run_dir.mkdir()
    pd.DataFrame(
        {
            "rebalance_date": ["20200101"],
            "entry_date": ["20200102"],
            "symbol": ["AAA"],
            "weight": [0.5],
            "side": ["long"],
        }
    ).to_csv(run_dir / "positions_by_rebalance.csv", index=False)
    pd.DataFrame(
        {
            "trade_date": ["20200102", "20200103"],
            "symbol": ["AAA", "AAA"],
            "close": [10.0, 10.0],
            "amount": [100_000_000.0, 100_000_000.0],
            "medadv20_amount": [100_000_000.0, 100_000_000.0],
            "is_tradable": [True, True],
            "is_buy_tradable": [True, True],
            "is_sell_tradable": [True, True],
        }
    ).to_csv(tmp_path / "pricing.csv", index=False)
    (run_dir / "config.used.yml").write_text(
        yaml.safe_dump(
            {
                "market": "a_share",
                "data": {"price_col": "close"},
                "backtest": {
                    "transaction_cost_bps": 0,
                    "trading_days_per_year": 252,
                    "tradable_col": "is_tradable",
                    "execution_sim": {
                        "enabled": True,
                        "portfolio_values": [1_000_000],
                        "participation_rate": 0.05,
                        "liquidity_cols": ["medadv20_amount"],
                        "buy_max_days": 1,
                        "sell_max_days": 1,
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps({"positions": {"by_rebalance_file": "positions_by_rebalance.csv"}}),
        encoding="utf-8",
    )

    payload = capacity_report.run(
        argparse.Namespace(
            run_dir=str(run_dir),
            config=None,
            positions_file=None,
            pricing_file=str(tmp_path / "pricing.csv"),
            portfolio_value=None,
            participation_rate=None,
            liquidity_col=None,
            primary_participation_rate=None,
            threshold_profile="neutral",
            output_dir=None,
            output_csv=None,
            output_json=None,
            market=None,
            log_level="INFO",
        )
    )

    row = payload["metrics_by_grid"][0]
    assert row["avg_cash_weight"] == pytest.approx(0.5)
    assert row["avg_target_cash_weight"] == pytest.approx(0.5)
    assert row["avg_execution_shortfall_cash_weight"] == pytest.approx(0.0)
    assert row["passed"] is True
    assert payload["status"] == "passed"
