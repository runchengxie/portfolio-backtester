"""Generic paired execution-metric comparison helpers."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


DEFAULT_EXECUTION_COMPARISON_METRICS: tuple[str, ...] = (
    "total_return",
    "annualized_return",
    "mean_daily_net_return",
    "annualized_volatility",
    "annualized_sharpe_zero_rf",
    "max_drawdown",
    "total_transaction_cost",
    "total_traded_notional",
    "mean_cash_weight",
)

_RESERVED_OUTPUT_COLUMNS = {
    "horizon",
    "single_side_cost_bps",
    "metric",
    "delta",
}


def compare_paired_execution_metrics(
    metrics: pd.DataFrame,
    *,
    baseline_variant: str,
    challenger_variant: str,
    baseline_output_column: str = "baseline",
    challenger_output_column: str = "challenger",
    value_columns: Sequence[str] = DEFAULT_EXECUTION_COMPARISON_METRICS,
) -> pd.DataFrame:
    """Return challenger-minus-baseline deltas for every execution cell.

    The helper deliberately accepts caller-supplied variant identities and output
    labels.  Portfolio code therefore owns the generic backtest comparison while
    strategy-specific names remain in the application layer.
    """

    if not baseline_variant.strip() or not challenger_variant.strip():
        raise ValueError("paired execution variants must be non-empty")
    if baseline_variant == challenger_variant:
        raise ValueError("paired execution variants must be distinct")
    if not baseline_output_column.strip() or not challenger_output_column.strip():
        raise ValueError("paired execution output columns must be non-empty")
    if baseline_output_column == challenger_output_column:
        raise ValueError("paired execution output columns must be distinct")
    if {
        baseline_output_column,
        challenger_output_column,
    } & _RESERVED_OUTPUT_COLUMNS:
        raise ValueError("paired execution output columns collide with reserved columns")

    compared_columns = tuple(value_columns)
    if not compared_columns or len(set(compared_columns)) != len(compared_columns):
        raise ValueError("paired execution value columns must be non-empty and unique")

    required = {
        "horizon",
        "single_side_cost_bps",
        "variant",
        *compared_columns,
    }
    missing = sorted(required - set(metrics.columns))
    if missing:
        raise ValueError(f"execution metrics are missing columns: {missing}")

    rows: list[dict[str, object]] = []
    expected_variants = {baseline_variant, challenger_variant}
    for raw_key, group in metrics.groupby(["horizon", "single_side_cost_bps"], sort=True):
        horizon, cost = raw_key
        indexed = group.set_index("variant")
        if len(indexed) != 2 or set(indexed.index) != expected_variants:
            raise ValueError("execution metrics do not contain exactly one paired variant row")
        for metric in compared_columns:
            baseline = float(indexed.at[baseline_variant, metric])
            challenger = float(indexed.at[challenger_variant, metric])
            rows.append(
                {
                    "horizon": int(horizon),
                    "single_side_cost_bps": float(cost),
                    "metric": metric,
                    baseline_output_column: baseline,
                    challenger_output_column: challenger,
                    "delta": challenger - baseline,
                }
            )

    return pd.DataFrame(
        rows,
        columns=[
            "horizon",
            "single_side_cost_bps",
            "metric",
            baseline_output_column,
            challenger_output_column,
            "delta",
        ],
    )


__all__ = [
    "DEFAULT_EXECUTION_COMPARISON_METRICS",
    "compare_paired_execution_metrics",
]
