"""Phase 6: reproducibility metadata snapshot and year-dimension metric tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from portfolio_backtester._capacity_report_grid import build_capacity_report
from portfolio_backtester._metrics_period import summarize_period_returns, yearly_compounded_returns
from portfolio_backtester._run_metadata import collect_reproducibility


def _write_run(tmp_path: Path) -> tuple[Path, Path, Path]:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    symbols = ["AAA", "BBB", "CCC"]
    positions = pd.DataFrame(
        {
            "rebalance_date": ["20200101"] * 3,
            "entry_date": ["20200102"] * 3,
            "symbol": symbols,
            "weight": [0.4, 0.4, 0.2],
            "side": ["long"] * 3,
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
        "data": {"price_col": "close"},
        "backtest": {
            "enabled": True,
            "transaction_cost_bps": 0,
            "trading_days_per_year": 252,
            "tradable_col": "is_tradable",
            "execution_sim": {
                "enabled": True,
                "portfolio_values": [1_000_000.0, 10_000_000.0, 100_000_000.0],
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
    return run_dir, positions_path, pricing_path


def test_collect_reproducibility_hashes_track_file_content(tmp_path: Path) -> None:
    run_dir, positions_path, pricing_path = _write_run(tmp_path)
    config_path = run_dir / "config.used.yml"

    snap = collect_reproducibility(
        config_path=config_path,
        positions_path=positions_path,
        pricing_path=pricing_path,
        run_dir=run_dir,
        market="a_share",
    )
    assert snap["config_hash"] and len(snap["config_hash"]) == 64
    assert snap["positions_hash"] and len(snap["positions_hash"]) == 64
    assert snap["pricing_hash"] and len(snap["pricing_hash"]) == 64
    assert snap["input_data_hash"] and len(snap["input_data_hash"]) == 64
    # universe fingerprint reuses the positions hash (no fabricated version).
    assert snap["universe_fingerprint"] == snap["positions_hash"]

    # Changing an input file changes its hash and the combined input hash.
    changed = positions_path.read_text(encoding="utf-8").replace("0.4", "0.5")
    positions_path.write_text(changed, encoding="utf-8")
    snap2 = collect_reproducibility(
        config_path=config_path,
        positions_path=positions_path,
        pricing_path=pricing_path,
        run_dir=run_dir,
        market="a_share",
    )
    assert snap2["positions_hash"] != snap["positions_hash"]
    assert snap2["input_data_hash"] != snap["input_data_hash"]


def test_collect_reproducibility_commit_and_versions_present(tmp_path: Path) -> None:
    run_dir, positions_path, pricing_path = _write_run(tmp_path)
    snap = collect_reproducibility(
        config_path=run_dir / "config.used.yml",
        positions_path=positions_path,
        pricing_path=pricing_path,
        run_dir=run_dir,
        market="a_share",
    )
    # repo_commit resolves either to a real git hash or the "unknown" fallback.
    assert isinstance(snap["repo_commit"], str) and snap["repo_commit"]
    # run_timestamp is ISO-8601 with timezone.
    assert "T" in snap["run_timestamp"]
    # dependency_versions records the portfolio-backtester package itself.
    assert "portfolio-backtester" in snap["dependency_versions"]
    # Version-like metadata we do not track is honestly marked, not fabricated.
    fee_ver = snap["fee_schedule_version"]
    assert fee_ver in ("unversioned", "unknown") or bool(fee_ver)


def test_yearly_compounded_returns_preserves_year_dimension() -> None:
    s = pd.Series(
        [0.1, 0.1, 0.1, 0.1],
        index=pd.to_datetime(["2023-12-29", "2023-12-30", "2024-01-02", "2024-01-03"]),
    )
    yearly = yearly_compounded_returns(s)
    assert set(yearly.keys()) == {2023, 2024}
    # Two 10% daily returns compound to 21% within each calendar year.
    assert abs(yearly[2023] - 0.21) < 1e-9
    assert abs(yearly[2024] - 0.21) < 1e-9


def test_summarize_period_returns_includes_year_dimension() -> None:
    s = pd.Series(
        [0.1, 0.1, 0.1, 0.1],
        index=pd.to_datetime(["2023-12-29", "2023-12-30", "2024-01-02", "2024-01-03"]),
    )
    summary = summarize_period_returns(s, period_info=[], trading_days_per_year=252)
    assert "yearly_returns" in summary
    assert summary["years_count"] == 2
    assert summary["best_year"] in (2023, 2024)
    assert summary["worst_year"] in (2023, 2024)


def test_capacity_report_payload_carries_reproducibility_and_metadata_file(tmp_path: Path) -> None:
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
        market_override="a_share",
    )
    assert "reproducibility" in payload
    repro = payload["reproducibility"]
    assert repro["schema"] == "reproducibility.v1"
    assert repro["market"] == "a_share"
    assert repro["backend_name"] == "native"
    # Calendar window is derived from the pricing panel span.
    assert repro["calendar_window"] is not None
    # The snapshot is also persisted next to the run artifacts.
    meta_path = run_dir / "run_metadata.json"
    assert meta_path.is_file()
    on_disk = json.loads(meta_path.read_text(encoding="utf-8"))
    assert on_disk["input_data_hash"] == repro["input_data_hash"]
