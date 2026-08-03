"""Execution model builders, describers, and L2 slippage helper."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from typing import Any, cast

from ._execution_models import (
    BpsCostModel,
    BpsSlippageModel,
    CostModel,
    DetailedTradeFeeModel,
    EntryPolicy,
    ExecutionModel,
    ExitFallbackPolicy,
    ExitPolicy,
    ExitPricePolicy,
    NoCostModel,
    NoSlippageModel,
    ParticipationSlippageModel,
    SelectionConstraints,
    SideBpsCostModel,
    SlippageModel,
)
from .execution_calendar import (
    MARKET_CALENDAR,
    coerce_date_set,
    normalize_execution_calendar,
)


def _coerce_non_negative_float(value: Any, *, label: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number < 0:
        raise ValueError(f"{label} must be >= 0.")
    return number


def _coerce_positive_float(value: Any, *, label: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise ValueError(f"{label} must be > 0.")
    return number


def _get_cost_alias(cfg: Mapping, *names: str, default: object = None) -> object:
    for name in names:
        if name in cfg:
            return cfg[name]
    return default


def _build_detailed_trade_fee_model(cost_cfg: Mapping) -> DetailedTradeFeeModel:
    base_commission = _get_cost_alias(cost_cfg, "commission_bps", default=2.5)
    base_slippage = _get_cost_alias(cost_cfg, "slippage_bps", default=10.0)
    return DetailedTradeFeeModel(
        buy_commission_bps=_coerce_non_negative_float(
            _get_cost_alias(cost_cfg, "buy_commission_bps", default=base_commission),
            label="cost_model.buy_commission_bps",
        ),
        sell_commission_bps=_coerce_non_negative_float(
            _get_cost_alias(cost_cfg, "sell_commission_bps", default=base_commission),
            label="cost_model.sell_commission_bps",
        ),
        sell_stamp_duty_bps=_coerce_non_negative_float(
            _get_cost_alias(cost_cfg, "sell_stamp_duty_bps", "stamp_tax_sell_bps", default=5.0),
            label="cost_model.sell_stamp_duty_bps",
        ),
        transfer_fee_bps=_coerce_non_negative_float(
            _get_cost_alias(cost_cfg, "transfer_fee_bps", default=0.1),
            label="cost_model.transfer_fee_bps",
        ),
        min_commission=_coerce_non_negative_float(
            _get_cost_alias(cost_cfg, "min_commission", "min_commission_cny", default=5.0),
            label="cost_model.min_commission",
        ),
        buy_slippage_bps=_coerce_non_negative_float(
            _get_cost_alias(cost_cfg, "buy_slippage_bps", default=base_slippage),
            label="cost_model.buy_slippage_bps",
        ),
        sell_slippage_bps=_coerce_non_negative_float(
            _get_cost_alias(cost_cfg, "sell_slippage_bps", default=base_slippage),
            label="cost_model.sell_slippage_bps",
        ),
        portfolio_value=_coerce_positive_float(
            _get_cost_alias(cost_cfg, "portfolio_value", default=1_000_000.0),
            label="cost_model.portfolio_value",
        ),
    )


def build_cost_model(cost_cfg: Mapping | None, default_bps: float) -> CostModel:
    if cost_cfg is None:
        return BpsCostModel(float(default_bps))
    if not isinstance(cost_cfg, Mapping):
        name = str(cost_cfg).strip().lower()
        if name in {"none", "zero", "off"}:
            return NoCostModel()
        return BpsCostModel(float(default_bps))

    name = str(cost_cfg.get("name", "bps")).strip().lower()
    if name in {"none", "zero", "off"}:
        return NoCostModel()
    if name in {"bps", "bp", "basis"}:
        bps = cost_cfg.get("bps", default_bps)
        round_trip = bool(cost_cfg.get("round_trip", True))
        return BpsCostModel(float(bps), round_trip=round_trip)
    if name in {"detailed", "detailed_fee", "trade_fee", "a_share_detailed"}:
        return _build_detailed_trade_fee_model(cost_cfg)
    if name in {"side_bps", "fee_schedule", "fees"}:
        base_bps = float(cost_cfg.get("bps", default_bps))
        return SideBpsCostModel(
            long_entry_bps=_coerce_non_negative_float(
                cost_cfg.get("buy_bps", cost_cfg.get("long_entry_bps", base_bps)),
                label="cost_model.long_entry_bps",
            ),
            long_exit_bps=_coerce_non_negative_float(
                cost_cfg.get("sell_bps", cost_cfg.get("long_exit_bps", base_bps)),
                label="cost_model.long_exit_bps",
            ),
            short_entry_bps=_coerce_non_negative_float(
                cost_cfg.get(
                    "short_entry_bps",
                    cost_cfg.get("short_open_bps", base_bps),
                ),
                label="cost_model.short_entry_bps",
            ),
            short_exit_bps=_coerce_non_negative_float(
                cost_cfg.get(
                    "short_exit_bps",
                    cost_cfg.get("short_close_bps", base_bps),
                ),
                label="cost_model.short_exit_bps",
            ),
            short_borrow_bps_per_day=_coerce_non_negative_float(
                cost_cfg.get(
                    "short_borrow_bps_per_day",
                    cost_cfg.get("borrow_bps_per_day", 0.0),
                ),
                label="cost_model.short_borrow_bps_per_day",
            ),
        )
    raise ValueError(f"Unsupported cost model: {name}")


def build_slippage_model(slippage_cfg: Mapping | None) -> SlippageModel:
    if slippage_cfg is None:
        return NoSlippageModel()
    if not isinstance(slippage_cfg, Mapping):
        name = str(slippage_cfg).strip().lower()
        if name in {"", "none", "zero", "off"}:
            return NoSlippageModel()
        if name in {"bps", "bp", "basis"}:
            return BpsSlippageModel(0.0)
        raise ValueError(f"Unsupported slippage model: {name}")

    name = str(slippage_cfg.get("name", "none")).strip().lower()
    if name in {"", "none", "zero", "off"}:
        return NoSlippageModel()
    if name in {"bps", "bp", "basis"}:
        return BpsSlippageModel(
            _coerce_non_negative_float(
                slippage_cfg.get("bps", slippage_cfg.get("base_bps", 0.0)),
                label="slippage_model.bps",
            )
        )
    if name in {"participation", "turnover_ratio", "adv_ratio", "impact"}:
        amount_col = str(slippage_cfg.get("amount_col", "amount")).strip() or "amount"
        power = _coerce_positive_float(
            slippage_cfg.get("power", 0.5),
            label="slippage_model.power",
        )
        max_participation = slippage_cfg.get("max_participation")
        if max_participation is not None:
            max_participation = _coerce_positive_float(
                max_participation,
                label="slippage_model.max_participation",
            )
        return ParticipationSlippageModel(
            base_bps=_coerce_non_negative_float(
                slippage_cfg.get("base_bps", 0.0),
                label="slippage_model.base_bps",
            ),
            impact_bps=_coerce_non_negative_float(
                slippage_cfg.get("impact_bps", 0.0),
                label="slippage_model.impact_bps",
            ),
            amount_col=amount_col,
            portfolio_value=_coerce_positive_float(
                slippage_cfg.get("portfolio_value", 1_000_000.0),
                label="slippage_model.portfolio_value",
            ),
            power=power,
            max_participation=max_participation,
        )
    raise ValueError(f"Unsupported slippage model: {name}")


def build_entry_policy(
    entry_cfg: Mapping | None,
    *,
    default_price_col: str = "close",
) -> EntryPolicy:
    if entry_cfg is None:
        return EntryPolicy(str(default_price_col))
    if not isinstance(entry_cfg, Mapping):
        price_col = str(entry_cfg).strip() or str(default_price_col)
        return EntryPolicy(price_col)

    price_col = (
        entry_cfg.get("price_col")
        or entry_cfg.get("column")
        or entry_cfg.get("price")
        or default_price_col
    )
    price_col = str(price_col).strip()
    if not price_col:
        raise ValueError("entry_policy.price_col cannot be empty.")
    return EntryPolicy(price_col)


def build_selection_constraints(
    constraints_cfg: Mapping | None,
) -> SelectionConstraints:
    if constraints_cfg is None:
        return SelectionConstraints()
    if not isinstance(constraints_cfg, Mapping):
        raise ValueError("execution.constraints must be a mapping.")

    min_price = constraints_cfg.get("min_price")
    if min_price is not None:
        min_price = _coerce_non_negative_float(min_price, label="constraints.min_price")
    min_amount = constraints_cfg.get("min_amount")
    if min_amount is not None:
        min_amount = _coerce_non_negative_float(min_amount, label="constraints.min_amount")
    amount_col = str(constraints_cfg.get("amount_col", "amount")).strip() or "amount"
    return SelectionConstraints(
        min_price=min_price,
        min_amount=min_amount,
        amount_col=amount_col,
    )


def build_exit_policy(
    exit_cfg: Mapping | None,
    default_price: ExitPricePolicy,
    default_fallback: ExitFallbackPolicy,
    *,
    default_price_col: str = "close",
) -> ExitPolicy:
    if exit_cfg is None:
        return ExitPolicy(default_price, default_fallback, str(default_price_col))
    if not isinstance(exit_cfg, Mapping):
        return ExitPolicy(default_price, default_fallback, str(default_price_col))

    price = exit_cfg.get("price") or exit_cfg.get("price_policy") or default_price
    fallback = exit_cfg.get("fallback") or exit_cfg.get("fallback_policy") or default_fallback
    price_col = exit_cfg.get("price_col") or exit_cfg.get("column") or default_price_col
    price = str(price).strip().lower()
    fallback = str(fallback).strip().lower()
    price_col = str(price_col).strip()
    if price not in {"strict", "ffill", "delay"}:
        raise ValueError("exit_policy.price must be one of: strict, ffill, delay.")
    if fallback not in {"ffill", "none"}:
        raise ValueError("exit_policy.fallback must be one of: ffill, none.")
    if not price_col:
        raise ValueError("exit_policy.price_col cannot be empty.")
    return ExitPolicy(cast(ExitPricePolicy, price), cast(ExitFallbackPolicy, fallback), price_col)


def build_execution_model(
    execution_cfg: Mapping | None,
    *,
    default_cost_bps: float,
    default_exit_price_policy: ExitPricePolicy,
    default_exit_fallback_policy: ExitFallbackPolicy,
    default_price_col: str = "close",
    default_entry_price_col: str | None = None,
    default_exit_price_col: str | None = None,
) -> ExecutionModel:
    cost_cfg = None
    exit_cfg = None
    slippage_cfg = None
    entry_cfg = None
    constraints_cfg = None
    calendar = MARKET_CALENDAR
    calendar_open_dates: tuple[pd.Timestamp, ...] = ()
    calendar_closed_dates: tuple[pd.Timestamp, ...] = ()
    if isinstance(execution_cfg, Mapping):
        cost_cfg = execution_cfg.get("cost_model") or execution_cfg.get("cost")
        slippage_cfg = execution_cfg.get("slippage_model") or execution_cfg.get("slippage")
        exit_cfg = execution_cfg.get("exit_policy") or execution_cfg.get("exit")
        entry_cfg = execution_cfg.get("entry_policy") or execution_cfg.get("entry")
        constraints_cfg = execution_cfg.get("constraints") or execution_cfg.get("selection")
        calendar = normalize_execution_calendar(execution_cfg.get("calendar"))
        calendar_open_dates = coerce_date_set(
            execution_cfg.get("open_dates")
            or execution_cfg.get("calendar_open_dates")
            or execution_cfg.get("stock_connect_open_dates")
        )
        calendar_closed_dates = coerce_date_set(
            execution_cfg.get("closed_dates")
            or execution_cfg.get("calendar_closed_dates")
            or execution_cfg.get("stock_connect_closed_dates")
        )
    entry_policy = build_entry_policy(
        entry_cfg,
        default_price_col=default_entry_price_col or default_price_col,
    )
    cost_model = build_cost_model(cost_cfg, default_cost_bps)
    slippage_model = build_slippage_model(slippage_cfg)
    exit_policy = build_exit_policy(
        exit_cfg,
        default_price=default_exit_price_policy,
        default_fallback=default_exit_fallback_policy,
        default_price_col=default_exit_price_col or default_price_col,
    )
    selection_constraints = build_selection_constraints(constraints_cfg)
    return ExecutionModel(
        cost_model=cost_model,
        slippage_model=slippage_model,
        exit_policy=exit_policy,
        entry_policy=entry_policy,
        selection_constraints=selection_constraints,
        calendar=calendar,
        calendar_open_dates=calendar_open_dates,
        calendar_closed_dates=calendar_closed_dates,
    )


def describe_cost_model(cost_model: CostModel) -> dict:
    if isinstance(cost_model, BpsCostModel):
        return {
            "name": "bps",
            "bps": float(cost_model.bps),
            "round_trip": bool(cost_model.round_trip),
        }
    if isinstance(cost_model, DetailedTradeFeeModel):
        return {
            "name": "detailed",
            "buy_commission_bps": float(cost_model.buy_commission_bps),
            "sell_commission_bps": float(cost_model.sell_commission_bps),
            "sell_stamp_duty_bps": float(cost_model.sell_stamp_duty_bps),
            "transfer_fee_bps": float(cost_model.transfer_fee_bps),
            "min_commission": float(cost_model.min_commission),
            "buy_slippage_bps": float(cost_model.buy_slippage_bps),
            "sell_slippage_bps": float(cost_model.sell_slippage_bps),
            "portfolio_value": float(cost_model.portfolio_value),
        }
    if isinstance(cost_model, SideBpsCostModel):
        return {
            "name": "side_bps",
            "long_entry_bps": float(cost_model.long_entry_bps),
            "long_exit_bps": float(cost_model.long_exit_bps),
            "short_entry_bps": float(cost_model.short_entry_bps),
            "short_exit_bps": float(cost_model.short_exit_bps),
            "short_borrow_bps_per_day": float(cost_model.short_borrow_bps_per_day),
        }
    if isinstance(cost_model, NoCostModel):
        return {"name": "none"}
    return {"name": cost_model.__class__.__name__}


def describe_slippage_model(slippage_model: SlippageModel) -> dict:
    if isinstance(slippage_model, NoSlippageModel):
        return {"name": "none"}
    if isinstance(slippage_model, BpsSlippageModel):
        return {"name": "bps", "bps": float(slippage_model.bps)}
    if isinstance(slippage_model, ParticipationSlippageModel):
        return {
            "name": "participation",
            "base_bps": float(slippage_model.base_bps),
            "impact_bps": float(slippage_model.impact_bps),
            "amount_col": slippage_model.amount_col,
            "portfolio_value": float(slippage_model.portfolio_value),
            "power": float(slippage_model.power),
            "max_participation": (
                float(slippage_model.max_participation)
                if slippage_model.max_participation is not None
                else None
            ),
        }
    return {"name": slippage_model.__class__.__name__}


def describe_selection_constraints(constraints: SelectionConstraints) -> dict:
    return {
        "min_price": (float(constraints.min_price) if constraints.min_price is not None else None),
        "min_amount": (
            float(constraints.min_amount) if constraints.min_amount is not None else None
        ),
        "amount_col": constraints.amount_col,
    }


def required_pricing_columns(model: ExecutionModel) -> set[str]:
    columns = {
        str(model.entry_policy.price_col).strip(),
        str(model.exit_policy.price_col).strip(),
    }
    if model.selection_constraints.min_amount is not None:
        columns.add(str(model.selection_constraints.amount_col).strip())
    if isinstance(model.slippage_model, ParticipationSlippageModel):
        columns.add(str(model.slippage_model.amount_col).strip())
    return {column for column in columns if column}


def describe_execution_model(model: ExecutionModel) -> dict:
    return {
        "calendar": model.calendar,
        "calendar_open_dates": [date.strftime("%Y%m%d") for date in model.calendar_open_dates],
        "calendar_closed_dates": [date.strftime("%Y%m%d") for date in model.calendar_closed_dates],
        "cost_model": describe_cost_model(model.cost_model),
        "slippage_model": describe_slippage_model(model.slippage_model),
        "entry_policy": {
            "price_col": model.entry_policy.price_col,
        },
        "exit_policy": {
            "price_policy": model.exit_policy.price_policy,
            "fallback_policy": model.exit_policy.fallback_policy,
            "price_col": model.exit_policy.price_col,
        },
        "constraints": describe_selection_constraints(model.selection_constraints),
    }


# L2 price-tiered slippage (half-spread bps from A-share order-book data).
_L2_HALF_SPREAD_TABLE = {(0, 10): 8.6, (10, 20): 4.4, (20, 30): 3.5, (30, float("inf")): 3.5}


def l2_price_tiered_slippage(close_price: float, *, side: str = "buy") -> float:
    """Half-spread slippage (bps) from A-share L2 order-book data per price tier."""
    base = 6.0
    for (lo, hi), hs in _L2_HALF_SPREAD_TABLE.items():
        if lo <= close_price < hi:
            base = hs
            break
    return base + 2.0 if str(side).strip().lower() == "sell" else base
