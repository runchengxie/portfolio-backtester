from __future__ import annotations

import json
import warnings
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pandas as pd
import pytest

from portfolio_backtester import BacktestSpec, GroupCap, StrategySpec, backtest_topk, run_backtest
from portfolio_backtester.execution import (
    ExecutionModel,
    SideBpsCostModel,
    build_execution_model,
    describe_execution_model,
)


class _CustomCostModel:
    def cost(self, *args, **kwargs) -> float:
        return 0.0


def _golden_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": pd.to_datetime(
                ["2020-01-01"] * 4 + ["2020-01-02"] * 4 + ["2020-01-03"] * 4
            ),
            "symbol": ["A", "B", "C", "D"] * 3,
            "pred": [4.0, 3.0, 2.0, 1.0, 3.0, 4.0, 2.0, 1.0, 3.0, 4.0, 2.0, 1.0],
            "close": [
                100.0,
                100.0,
                100.0,
                100.0,
                110.0,
                90.0,
                105.0,
                100.0,
                121.0,
                81.0,
                110.25,
                100.0,
            ],
            "industry": ["X", "X", "Y", "Y"] * 3,
            "amount": [1_000.0] * 12,
        }
    )


def _execution_model() -> ExecutionModel:
    return build_execution_model(
        {
            "cost": {"name": "side_bps", "buy_bps": 10, "sell_bps": 5},
            "slippage": {"name": "bps", "bps": 2},
            "constraints": {
                "min_price": 1.0,
                "min_amount": 100.0,
                "amount_col": "amount",
            },
            "entry": {"price_col": "close"},
            "exit": {"price": "strict", "fallback": "ffill", "price_col": "close"},
            "calendar": "market",
        },
        default_cost_bps=0.0,
        default_exit_price_policy="strict",
        default_exit_fallback_policy="ffill",
        default_price_col="close",
    )


def _golden_spec() -> BacktestSpec:
    return BacktestSpec(
        strategy=StrategySpec(
            name="industry-aware",
            type="topk_buffered_long_only",
            score_col="pred",
            top_k=2,
            buffer_exit=0,
            buffer_entry=0,
            weighting="equal",
            long_only=True,
            group_cap=GroupCap(column="industry", max_names=1),
        ),
        execution=_execution_model(),
        rebalance_dates=tuple(pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])),
        shift_days=0,
        trading_days_per_year=252,
    )


def _assert_same_result(left, right) -> None:
    assert left is not None
    assert right is not None
    left_stats, left_net, left_gross, left_turnover, left_periods = left
    right_stats, right_net, right_gross, right_turnover, right_periods = right
    assert left_stats.keys() == right_stats.keys()
    for key in left_stats:
        left_value = left_stats[key]
        right_value = right_stats[key]
        if isinstance(left_value, (float, np.floating)) and np.isnan(left_value):
            assert isinstance(right_value, (float, np.floating)) and np.isnan(right_value)
        elif isinstance(left_value, (float, np.floating)):
            assert right_value == pytest.approx(left_value)
        else:
            assert left_value == right_value
    pd.testing.assert_series_equal(left_net, right_net)
    pd.testing.assert_series_equal(left_gross, right_gross)
    pd.testing.assert_series_equal(left_turnover, right_turnover)
    assert left_periods == right_periods


def test_backtest_spec_is_frozen_and_json_serializable() -> None:
    spec = replace(
        _golden_spec(),
        liquidity_floor_col="amount",
        liquidity_floor_quantile=0.25,
        max_turnover_per_rebalance=0.4,
        selection_tiebreak_col="amount",
        selection_score_bucket_size=0.01,
        selection_score_margin=0.02,
        selection_score_margin_col="candidate_relevance",
        selection_score_margin_rank_limit=10,
        selection_min_score=0.25,
        max_new_names_per_rebalance=3,
        max_new_names_shortfall_policy="carry",
        max_positive_names=10,
    )

    mapping = spec.to_mapping()
    encoded = json.dumps(mapping, allow_nan=False, sort_keys=True)
    restored = BacktestSpec.from_mapping(json.loads(encoded))

    assert restored == spec
    assert restored.strategy.group_cap == GroupCap(column="industry", max_names=1)
    assert restored.execution is not None
    assert isinstance(restored.execution.cost_model, SideBpsCostModel)
    assert restored.selection_score_margin_col == "candidate_relevance"
    assert restored.max_new_names_shortfall_policy == "carry"
    assert restored.max_positive_names == 10
    assert describe_execution_model(restored.execution) == describe_execution_model(spec.execution)
    with pytest.raises(FrozenInstanceError):
        spec.rank_offset = 1  # type: ignore[misc]


def test_backtest_spec_rejects_unknown_schema_version() -> None:
    mapping = _golden_spec().to_mapping()
    mapping["schema_version"] = 2

    with pytest.raises(ValueError, match="Unsupported BacktestSpec schema version: 2"):
        BacktestSpec.from_mapping(mapping)


def test_backtest_spec_reads_schema_v1_mapping_without_additive_controls() -> None:
    mapping = _golden_spec().to_mapping()
    for field in (
        "selection_score_margin_col",
        "max_new_names_shortfall_policy",
        "max_positive_names",
    ):
        mapping.pop(field)

    restored = BacktestSpec.from_mapping(mapping)

    assert restored.selection_score_margin_col is None
    assert restored.max_new_names_shortfall_policy == "legacy_concentrate"
    assert restored.max_positive_names is None


@pytest.mark.parametrize("invalid", [True, 1.5, "1"])
def test_backtest_spec_rejects_non_integer_new_name_budget(invalid: object) -> None:
    mapping = _golden_spec().to_mapping()
    mapping["max_new_names_per_rebalance"] = invalid

    with pytest.raises(ValueError, match="non-negative integer"):
        BacktestSpec.from_mapping(mapping)


def test_backtest_spec_rejects_unserializable_custom_execution_component() -> None:
    execution = replace(_execution_model(), cost_model=_CustomCostModel())
    spec = replace(_golden_spec(), execution=execution)

    with pytest.raises(TypeError, match="supports built-in execution models only"):
        spec.to_mapping()


def test_run_backtest_matches_golden_result_and_compatibility_facade() -> None:
    data = _golden_frame()
    spec = _golden_spec()

    result = run_backtest(data, spec)

    assert result is not None
    stats, net, gross, turnover, periods = result
    assert net.tolist() == pytest.approx([0.0738, -0.02597209302325578])
    assert gross.tolist() == pytest.approx([0.075, -0.025])
    assert turnover.tolist() == pytest.approx([1.0, 0.5116279069767442])
    assert stats["total_return"] == pytest.approx(0.045911166511628076)
    assert stats["avg_fee_drag"] == pytest.approx(0.0008837209302325581)
    assert stats["avg_slippage_drag"] == pytest.approx(0.00020232558139534885)
    assert [period["exit_date"] for period in periods] == list(
        pd.to_datetime(["2020-01-02", "2020-01-03"])
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        facade_result = backtest_topk(
            data,
            pred_col="pred",
            price_col="close",
            rebalance_dates=list(spec.rebalance_dates),
            top_k=2,
            shift_days=0,
            cost_bps=0.0,
            trading_days_per_year=252,
            group_col="industry",
            max_names_per_group=1,
            execution=spec.execution,
        )

    assert not [item for item in caught if issubclass(item.category, DeprecationWarning)]
    _assert_same_result(result, facade_result)


def test_run_backtest_accepts_separate_pricing_data() -> None:
    data = _golden_frame().query("not (trade_date == '2020-01-02' and symbol == 'A')")
    spec = _golden_spec()

    assert run_backtest(data, spec, pricing_data=_golden_frame()) is not None


def test_run_backtest_applies_score_threshold_and_new_name_limit() -> None:
    spec = replace(
        _golden_spec(),
        selection_min_score=2.5,
        max_new_names_per_rebalance=0,
    )

    result = run_backtest(_golden_frame(), spec)

    assert result is not None
    _, _, gross, _, _ = result
    assert gross.tolist() == pytest.approx([0.10, 0.10])
