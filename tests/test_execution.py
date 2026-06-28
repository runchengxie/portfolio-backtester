from __future__ import annotations

import pytest

from cstree.backtesting.execution import (
    BpsCostModel,
    DetailedTradeFeeModel,
    ExitPolicy,
    NoCostModel,
    ParticipationSlippageModel,
    SideBpsCostModel,
    build_cost_model,
    build_execution_model,
    build_exit_policy,
    required_pricing_columns,
)


@pytest.mark.parametrize("value", ["none", "off", "zero"])
def test_build_cost_model_string_aliases_disable_cost(value: str) -> None:
    model = build_cost_model(value, default_bps=15.0)

    assert isinstance(model, NoCostModel)


def test_build_cost_model_detailed_fee_schedule() -> None:
    model = build_cost_model(
        {
            "name": "detailed",
            "commission_bps": 1.0,
            "stamp_tax_sell_bps": 5.0,
            "transfer_fee_bps": 0.1,
            "min_commission_cny": 5.0,
            "buy_slippage_bps": 2.0,
            "sell_slippage_bps": 3.0,
            "portfolio_value": 10_000.0,
        },
        default_bps=0.0,
    )

    assert isinstance(model, DetailedTradeFeeModel)
    assert model.notional_cost(10_000.0, side="buy") == pytest.approx(7.1)
    assert model.notional_cost(10_000.0, side="sell") == pytest.approx(13.1)
    assert model.cost(
        1.0,
        is_initial=True,
        side="long",
        entry_turnover=1.0,
        exit_turnover=0.0,
        gross_exposure=1.0,
    ) == pytest.approx(0.00071)


@pytest.mark.parametrize("name", ["bps", "bp", "basis"])
def test_build_cost_model_mapping_aliases(name: str) -> None:
    model = build_cost_model({"name": name, "bps": 12, "round_trip": False}, default_bps=15.0)

    assert isinstance(model, BpsCostModel)
    assert model.bps == 12.0
    assert model.round_trip is False


def test_build_cost_model_unsupported_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported cost model: flat"):
        build_cost_model({"name": "flat"}, default_bps=15.0)


def test_build_exit_policy_supports_alias_keys() -> None:
    policy = build_exit_policy(
        {"price_policy": "delay", "fallback_policy": "none"},
        default_price="strict",
        default_fallback="ffill",
    )

    assert isinstance(policy, ExitPolicy)
    assert policy.price_policy == "delay"
    assert policy.fallback_policy == "none"


def test_build_exit_policy_invalid_value_raises() -> None:
    with pytest.raises(ValueError, match="exit_policy.price must be one of: strict, ffill, delay."):
        build_exit_policy({"price": "bad"}, default_price="strict", default_fallback="ffill")


def test_build_execution_model_supports_cost_and_exit_alias() -> None:
    model = build_execution_model(
        {
            "cost": {"name": "bp", "bps": 8, "round_trip": False},
            "exit": {"price": "delay", "fallback": "none"},
        },
        default_cost_bps=20.0,
        default_exit_price_policy="strict",
        default_exit_fallback_policy="ffill",
    )

    assert isinstance(model.cost_model, BpsCostModel)
    assert model.cost_model.bps == 8.0
    assert model.cost_model.round_trip is False
    assert model.exit_policy.price_policy == "delay"
    assert model.exit_policy.fallback_policy == "none"
    assert model.exit_policy.price_col == "close"


def test_build_execution_model_uses_defaults_when_empty() -> None:
    model = build_execution_model(
        None,
        default_cost_bps=20.0,
        default_exit_price_policy="ffill",
        default_exit_fallback_policy="ffill",
    )

    assert isinstance(model.cost_model, BpsCostModel)
    assert model.cost_model.bps == 20.0
    assert model.exit_policy.price_policy == "ffill"
    assert model.exit_policy.fallback_policy == "ffill"
    assert model.entry_policy.price_col == "close"


def test_build_execution_model_supports_slippage_entry_and_constraints() -> None:
    model = build_execution_model(
        {
            "cost": {
                "name": "side_bps",
                "buy_bps": 6,
                "sell_bps": 8,
                "short_borrow_bps_per_day": 1,
            },
            "slippage": {
                "name": "participation",
                "base_bps": 2,
                "impact_bps": 10,
                "amount_col": "amount",
                "portfolio_value": 500000,
            },
            "entry": {"price_col": "open"},
            "exit": {"price": "delay", "fallback": "none", "price_col": "close"},
            "constraints": {"min_price": 5, "min_amount": 100000, "amount_col": "amount"},
        },
        default_cost_bps=20.0,
        default_exit_price_policy="strict",
        default_exit_fallback_policy="ffill",
    )

    assert isinstance(model.cost_model, SideBpsCostModel)
    assert isinstance(model.slippage_model, ParticipationSlippageModel)
    assert model.entry_policy.price_col == "open"
    assert model.exit_policy.price_col == "close"
    assert model.selection_constraints.min_price == pytest.approx(5.0)
    assert model.selection_constraints.min_amount == pytest.approx(100000.0)
    assert required_pricing_columns(model) == {"open", "close", "amount"}
