import json
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

from cstree.backtesting import construction_grid
from cstree.backtesting.engine import backtest_topk


def _scored_data() -> pd.DataFrame:
    dates = pd.to_datetime(["2020-01-03", "2020-01-10", "2020-01-17", "2020-01-24"])
    symbols = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
    rows = []
    for d_idx, trade_date in enumerate(dates):
        for s_idx, symbol in enumerate(symbols):
            score = float(3 - s_idx + d_idx * 0.01)
            rows.append(
                {
                    "trade_date": trade_date,
                    "symbol": symbol,
                    "close": 100.0 + d_idx * (3 - s_idx) + s_idx,
                    "future_return": 0.02 * (3 - s_idx),
                    "signal_eval": score,
                    "signal_backtest": score,
                    "alt_signal": float(s_idx + d_idx * 0.01),
                    "sector_beta": float(s_idx),
                    "medadv20_amount": float(1_000_000 + s_idx * 100_000),
                    "is_tradable": True,
                }
            )
    return pd.DataFrame(rows)


def _write_summary(run_dir, scored_file):
    dates = [d.strftime("%Y%m%d") for d in sorted(_scored_data()["trade_date"].unique())]
    payload = {
        "run": {"output_dir": str(run_dir)},
        "data": {"min_symbols_per_date": 2, "price_col": "close"},
        "label": {"target_col": "future_return", "horizon_days": 5, "shift_days": 0},
        "eval": {
            "rebalance_frequency": "W",
            "rebalance_dates": dates,
            "scored_file": str(scored_file),
            "scored_signal_col": "signal_eval",
            "scored_signal_backtest_col": "signal_backtest",
        },
        "backtest": {
            "enabled": True,
            "rebalance_frequency": "W",
            "rebalance_dates": dates,
            "shift_days": 0,
            "trading_days_per_year": 252,
            "exit_price_policy": "strict",
            "exit_fallback_policy": "ffill",
            "tradable_col": "is_tradable",
        },
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
    return summary_path


def _write_weekly_benchmark(tmp_path):
    benchmark = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-10", "2020-01-17", "2020-01-24"]),
            "benchmark_return": [0.001, 0.001, 0.001],
        }
    )
    benchmark_path = tmp_path / "benchmark.csv"
    benchmark.to_csv(benchmark_path, index=False)
    return benchmark_path


def _args(config_path):
    return SimpleNamespace(
        config=str(config_path),
        output=None,
        output_json=None,
        log_level="INFO",
        backtest_topk_fn=backtest_topk,
    )


def test_construction_grid_reuses_existing_scores(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    scored_file = run_dir / "eval_scored.parquet"
    _scored_data().to_parquet(scored_file)
    summary_path = _write_summary(run_dir, scored_file)

    benchmark_path = _write_weekly_benchmark(tmp_path)

    cfg = {
        "construction_grid": {
            "summary_file": str(summary_path),
            "output_csv": str(tmp_path / "grid.csv"),
            "variants": [
                {
                    "name": "k1_equal",
                    "top_k": 1,
                    "cost_bps": 10,
                    "weighting": "equal",
                    "benchmark_name": "bench",
                    "benchmark_returns_file": str(benchmark_path),
                },
                {
                    "name": "neutralized",
                    "top_k": 2,
                    "cost_bps": 20,
                    "weighting": "signal",
                    "score_postprocess": {
                        "method": "neutralize",
                        "columns": ["sector_beta"],
                        "min_obs": 2,
                    },
                },
                {
                    "name": "dynamic_risk",
                    "top_k": 2,
                    "cost_bps": 15,
                    "weighting": "sqrt_liquidity",
                    "weighting_liquidity_col": "medadv20_amount",
                    "max_turnover_per_rebalance": 1.0,
                    "dynamic_ensemble": {
                        "signal_cols": ["signal_backtest", "alt_signal"],
                        "output_col": "__dyn_score",
                        "config": {
                            "min_history": 1,
                            "evaluation_window": 1,
                            "covariance_window": 1,
                            "min_icir": None,
                            "min_long_short_sharpe": None,
                            "min_stability": None,
                            "min_coverage_ratio": 0.0,
                            "min_signal_dispersion": 0.0,
                            "min_rank_ic_mean": None,
                            "min_direction_consistency": 0.0,
                            "stock_selection_count": 2,
                        },
                    },
                    "risk_penalty": {
                        "columns": ["sector_beta"],
                        "strength": 0.01,
                    },
                },
            ],
        }
    }
    config_path = tmp_path / "construction.yml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    rows = construction_grid.run(_args(config_path))

    assert {row["variant"] for row in rows} == {
        "k1_equal",
        "neutralized",
        "dynamic_risk",
    }
    assert {row["status"] for row in rows} == {"ok"}
    row = next(item for item in rows if item["variant"] == "k1_equal")
    assert row["information_ratio"] is not None
    assert row["backtest_avg_cost_drag"] is not None
    dynamic_row = next(item for item in rows if item["variant"] == "dynamic_risk")
    assert dynamic_row["dynamic_ensemble_active"] is True
    assert dynamic_row["risk_penalty_columns"] == "sector_beta"
    assert dynamic_row["weighting"] == "sqrt_liquidity"
    assert (tmp_path / "grid.csv").exists()


def test_construction_grid_accepts_rank_blend_score_postprocess(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    scored_file = run_dir / "eval_scored.parquet"
    _scored_data().to_parquet(scored_file)
    summary_path = _write_summary(run_dir, scored_file)

    cfg = {
        "construction_grid": {
            "summary_file": str(summary_path),
            "variants": [
                {
                    "name": "rank_blend",
                    "top_k": 2,
                    "cost_bps": 20,
                    "score_postprocess": {
                        "method": "rank_blend",
                        "columns": ["medadv20_amount"],
                        "strength": 0.05,
                    },
                },
            ],
        }
    }
    config_path = tmp_path / "construction.yml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    rows = construction_grid.run(_args(config_path))

    assert len(rows) == 1
    assert rows[0]["status"] == "ok"
    assert rows[0]["score_postprocess_method"] == "rank_blend"
    assert rows[0]["score_postprocess_columns"] == "medadv20_amount"


def test_construction_grid_accepts_explicit_rebalance_dates(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    scored_file = run_dir / "eval_scored.parquet"
    _scored_data().to_parquet(scored_file)
    summary_path = _write_summary(run_dir, scored_file)

    cfg = {
        "construction_grid": {
            "summary_file": str(summary_path),
            "rebalance_dates": ["20200103", "20200110"],
            "variants": [{"name": "k1_equal", "top_k": 1, "cost_bps": 0}],
        }
    }

    rows = construction_grid.build_construction_grid(
        cfg,
        config_dir=tmp_path,
        backtest_topk_fn=backtest_topk,
    )

    assert rows[0]["status"] == "ok"
    assert rows[0]["backtest_periods"] == 1


def test_construction_grid_compounds_daily_benchmark_returns(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    scored_file = run_dir / "eval_scored.parquet"
    _scored_data().to_parquet(scored_file)
    summary_path = _write_summary(run_dir, scored_file)

    benchmark = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(
                [
                    "2020-01-06",
                    "2020-01-10",
                    "2020-01-13",
                    "2020-01-17",
                    "2020-01-20",
                    "2020-01-24",
                ]
            ),
            "benchmark_return": [0.10, 0.0, 0.10, 0.0, 0.10, 0.0],
        }
    )
    benchmark_path = tmp_path / "benchmark_daily.csv"
    benchmark.to_csv(benchmark_path, index=False)

    cfg = {
        "construction_grid": {
            "summary_file": str(summary_path),
            "variants": [
                {
                    "name": "k1_equal",
                    "top_k": 1,
                    "cost_bps": 0,
                    "benchmark_returns_file": str(benchmark_path),
                },
            ],
        }
    }
    config_path = tmp_path / "construction.yml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    rows = construction_grid.run(_args(config_path))

    row = rows[0]
    assert row["status"] == "ok"
    assert row["active_total_return"] < row["backtest_total_return"]


def test_construction_grid_accepts_external_pricing_file(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    scored_file = run_dir / "eval_scored.parquet"
    scored = _scored_data()
    scored.to_parquet(scored_file)
    summary_path = _write_summary(run_dir, scored_file)

    pricing_file = tmp_path / "pricing.parquet"
    pricing = scored[["trade_date", "symbol", "close", "is_tradable"]].copy()
    pricing.to_parquet(pricing_file)

    cfg = {
        "construction_grid": {
            "summary_file": str(summary_path),
            "pricing_file": str(pricing_file),
            "variants": [{"name": "k2", "top_k": 2, "cost_bps": 0}],
        }
    }
    config_path = tmp_path / "construction.yml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    rows = construction_grid.run(_args(config_path))

    assert rows[0]["status"] == "ok"
    assert rows[0]["backtest_periods"] is not None


def test_construction_grid_passes_rank_offset_to_backtest(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    scored_file = run_dir / "eval_scored.parquet"
    _scored_data().to_parquet(scored_file)
    summary_path = _write_summary(run_dir, scored_file)

    cfg = {
        "construction_grid": {
            "summary_file": str(summary_path),
            "variants": [
                {"name": "top1", "top_k": 1, "cost_bps": 0},
                {"name": "rank2", "top_k": 1, "rank_offset": 1, "cost_bps": 0},
            ],
        }
    }
    config_path = tmp_path / "construction.yml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    rows = construction_grid.run(_args(config_path))

    by_name = {row["variant"]: row for row in rows}
    assert by_name["rank2"]["status"] == "ok"
    assert by_name["rank2"]["rank_offset"] == 1
    assert by_name["rank2"]["backtest_total_return"] < by_name["top1"]["backtest_total_return"]


def test_construction_grid_reads_yyyymmdd_benchmark_dates(tmp_path):
    benchmark = tmp_path / "benchmark.csv"
    benchmark.write_text(
        "trade_date,benchmark_return\n20200103,0.01\n2020-01-06,-0.02\n",
        encoding="utf-8",
    )

    series = construction_grid._read_returns_file(benchmark)

    assert list(series.index) == [pd.Timestamp("2020-01-03"), pd.Timestamp("2020-01-06")]
    assert series.iloc[0] == pytest.approx(0.01)


def test_construction_grid_fails_on_missing_required_column(tmp_path):
    scored_file = tmp_path / "bad.parquet"
    _scored_data().drop(columns=["close"]).to_parquet(scored_file)
    cfg = {
        "construction_grid": {
            "scored_file": str(scored_file),
            "variants": [{"name": "k1", "top_k": 1}],
        }
    }

    with pytest.raises(SystemExit, match="Missing required columns"):
        construction_grid.build_construction_grid(cfg, config_dir=tmp_path)


def test_construction_grid_inertia_selection_requires_switch_edge():
    rows = [
        {"variant": "old", "status": "ok", "information_ratio": 1.0},
        {"variant": "new", "status": "ok", "information_ratio": 1.05},
    ]

    kept = construction_grid.select_construction_variant_with_inertia(
        rows,
        previous_variant="old",
        objective_col="information_ratio",
        switch_penalty=0.10,
    )
    switched = construction_grid.select_construction_variant_with_inertia(
        rows,
        previous_variant="old",
        objective_col="information_ratio",
        switch_penalty=0.01,
    )

    assert kept["selected_variant"] == "old"
    assert kept["switched"] is False
    assert switched["selected_variant"] == "new"
    assert switched["switched"] is True


def test_construction_grid_writes_inertia_selection_report(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    scored_file = run_dir / "eval_scored.parquet"
    _scored_data().to_parquet(scored_file)
    summary_path = _write_summary(run_dir, scored_file)
    output_path = tmp_path / "selection.json"
    cfg = {
        "construction_grid": {
            "summary_file": str(summary_path),
            "rolling_selection": {
                "previous_variant": "k1",
                "objective_col": "backtest_sharpe",
                "switch_penalty": 9999.0,
                "output_json": str(output_path),
            },
            "variants": [
                {"name": "k1", "top_k": 1, "weighting": "equal"},
                {"name": "k2", "top_k": 2, "weighting": "equal"},
            ],
        }
    }
    config_path = tmp_path / "construction.yml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    construction_grid.run(_args(config_path))

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["artifact_type"] == "cstree.construction_grid_rolling_selection"
    assert payload["selected_variant"] == "k1"
