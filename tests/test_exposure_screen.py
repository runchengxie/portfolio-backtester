import json
from pathlib import Path

import pandas as pd

from portfolio_backtester import exposure_screen
from portfolio_backtester.exposure import _build_active_exposure_summary_table


def test_active_exposure_summary_matches_integer_rebalance_dates():
    style = pd.DataFrame(
        [
            {
                "rebalance_date": 20240726,
                "entry_date": 20240729,
                "factor": "momentum",
                "active_net_vs_equal": -0.4,
                "active_net_vs_cap": -0.8,
                "weight_coverage": 1.0,
                "source": "columns",
            }
        ]
    )
    industry = pd.DataFrame(
        [
            {
                "rebalance_date": 20240726,
                "entry_date": 20240729,
                "industry": "银行",
                "industry_col": "first_industry_name",
                "active_net_vs_equal_weight": -0.1,
                "active_net_vs_cap_weight": -0.21,
                "portfolio_net_weight": 0.0,
            }
        ]
    )

    active = _build_active_exposure_summary_table(style, industry)

    assert active.loc[0, "rebalance_date"] == "20240726"
    assert active.loc[0, "entry_date"] == "20240729"
    assert active.loc[0, "momentum_active_net_vs_cap"] == -0.8
    assert active.loc[0, "industry_top_1_name"] == "银行"
    assert active.loc[0, "industry_top_1_active"] == -0.21


def test_exposure_screen_reads_summary_and_flags_breaches(tmp_path: Path):
    active = tmp_path / "backtest_active_exposure_summary_oos.csv"
    row = {
        "rebalance_date": "20260529",
        "entry_date": "20260601",
        "size_active_net_vs_equal": 0.2,
        "size_active_net_vs_cap": -0.5,
        "size_weight_coverage": 1.0,
        "momentum_active_net_vs_equal": 1.2,
        "momentum_active_net_vs_cap": 0.1,
        "momentum_weight_coverage": 1.0,
        "industry_top_1_name": "Banks",
        "industry_top_1_active": 0.24,
        "industry_top_1_portfolio_net_weight": 0.4,
    }
    pd.DataFrame([row]).to_csv(active, index=False)
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps({"final_oos": {"backtest": {"exposure": {"active_summary_file": str(active)}}}}),
        encoding="utf-8",
    )

    payload = exposure_screen.build_exposure_screen(summary_file=summary)

    assert payload["status"] == "breached"
    breached = [check for check in payload["checks"] if check["status"] == "breached"]
    assert {check["check"] for check in breached} == {
        "style_active",
        "industry_active",
        "industry_weight",
    }


def test_exposure_screen_passes_when_latest_row_is_within_thresholds(tmp_path: Path):
    active = tmp_path / "active.csv"
    pd.DataFrame(
        [
            {
                "rebalance_date": "20260529",
                "entry_date": "20260601",
                "size_active_net_vs_equal": 0.2,
                "size_active_net_vs_cap": -0.5,
                "size_weight_coverage": 1.0,
                "beta_active_net_vs_equal": -0.1,
                "beta_active_net_vs_cap": 0.4,
                "beta_weight_coverage": 1.0,
                "industry_top_1_name": "Banks",
                "industry_top_1_active": 0.12,
                "industry_top_1_portfolio_net_weight": 0.2,
            }
        ]
    ).to_csv(active, index=False)

    payload = exposure_screen.build_exposure_screen(active_summary_file=active)

    assert payload["status"] == "passed"
    assert payload["breach_count"] == 0
