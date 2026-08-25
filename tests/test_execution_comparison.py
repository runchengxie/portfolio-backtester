from __future__ import annotations

import pandas as pd
import pytest

from portfolio_backtester.execution_comparison import (
    DEFAULT_EXECUTION_COMPARISON_METRICS,
    compare_paired_execution_metrics,
)


def _metrics() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for variant, offset in (("NUMERIC", 0.0), ("GPT_CODEX_SESSION", 0.02)):
        row: dict[str, object] = {
            "horizon": 1,
            "single_side_cost_bps": 20.0,
            "variant": variant,
        }
        for index, metric in enumerate(DEFAULT_EXECUTION_COMPARISON_METRICS, start=1):
            row[metric] = index / 100.0 + offset
        rows.append(row)
    return pd.DataFrame(rows)


def test_compare_paired_execution_metrics_preserves_strategy_app_output_shape() -> None:
    comparisons = compare_paired_execution_metrics(
        _metrics(),
        baseline_variant="NUMERIC",
        challenger_variant="GPT_CODEX_SESSION",
        baseline_output_column="numeric",
        challenger_output_column="gpt_codex_session",
    )

    assert comparisons.columns.tolist() == [
        "horizon",
        "single_side_cost_bps",
        "metric",
        "numeric",
        "gpt_codex_session",
        "delta",
    ]
    assert comparisons["metric"].tolist() == list(DEFAULT_EXECUTION_COMPARISON_METRICS)
    total_return = comparisons.loc[comparisons["metric"].eq("total_return")].iloc[0]
    assert total_return["numeric"] == pytest.approx(0.01)
    assert total_return["gpt_codex_session"] == pytest.approx(0.03)
    assert total_return["delta"] == pytest.approx(0.02)


def test_compare_paired_execution_metrics_rejects_duplicate_variant_row() -> None:
    metrics = pd.concat([_metrics(), _metrics().iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="exactly one paired variant row"):
        compare_paired_execution_metrics(
            metrics,
            baseline_variant="NUMERIC",
            challenger_variant="GPT_CODEX_SESSION",
        )
