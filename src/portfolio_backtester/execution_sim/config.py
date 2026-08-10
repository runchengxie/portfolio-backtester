from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

SELL_UNTIL_NEXT_REBALANCE = "until_next_rebalance"

__all__ = [
    "SELL_UNTIL_NEXT_REBALANCE",
    "ExecutionSimConfig",
    "build_execution_sim_config",
    "describe_execution_sim_config",
    "required_execution_sim_columns",
]


@dataclass(frozen=True)
class ExecutionSimConfig:
    enabled: bool = False
    portfolio_value: float = 1_000_000.0
    participation_rate: float = 0.05
    liquidity_cols: tuple[str, ...] = ("medadv20_amount", "amount")
    buy_max_days: int = 5
    sell_max_days: int | str = 10
    zero_fill_abort_days_buy: int | None = 5
    unfilled_buy_action: str = "keep_cash"
    unfilled_sell_action: str = "keep_position"
    # Phase 4 (市场规则与时间戳) opt-in 契约: 默认全部关闭, 保持现有固定对照场景基线.
    round_lot: int | None = None
    enforce_t1: bool = False
    enforce_price_limits: bool = False
    enforce_listing_status: bool = False
    limit_up_col: str | None = None
    limit_down_col: str | None = None
    listing_status_col: str | None = None
    lot_tolerance: float = 1e-6


def build_execution_sim_config(
    sim_cfg: object,
    *,
    default_portfolio_value: float = 1_000_000.0,
    default_liquidity_col: str = "medadv20_amount",
) -> ExecutionSimConfig:
    if sim_cfg is None:
        return ExecutionSimConfig(enabled=False)
    if isinstance(sim_cfg, bool):
        if not sim_cfg:
            return ExecutionSimConfig(enabled=False)
        sim_cfg = {"enabled": True}
    if not isinstance(sim_cfg, Mapping):
        raise ValueError("backtest.execution_sim must be a mapping or boolean.")

    enabled = bool(sim_cfg.get("enabled", False))
    if not enabled:
        return ExecutionSimConfig(enabled=False)

    portfolio_value = _coerce_positive_float(
        sim_cfg.get("portfolio_value", default_portfolio_value),
        label="execution_sim.portfolio_value",
    )
    participation_rate = _coerce_positive_float(
        sim_cfg.get("participation_rate", sim_cfg.get("participation", 0.05)),
        label="execution_sim.participation_rate",
    )
    liquidity_cols = _resolve_liquidity_cols(
        cast("Mapping[str, Any]", sim_cfg),
        default_liquidity_col=default_liquidity_col,
    )
    buy_max_days = _coerce_positive_int(
        sim_cfg.get("buy_max_days", 5),
        label="execution_sim.buy_max_days",
    )
    sell_max_days = _resolve_sell_max_days(sim_cfg.get("sell_max_days", 10))
    zero_fill_abort_days_buy_raw = sim_cfg.get("zero_fill_abort_days_buy", 5)
    if zero_fill_abort_days_buy_raw is None:
        zero_fill_abort_days_buy = None
    else:
        zero_fill_abort_days_buy = _coerce_positive_int(
            zero_fill_abort_days_buy_raw,
            label="execution_sim.zero_fill_abort_days_buy",
        )

    unfilled_buy_action = str(sim_cfg.get("unfilled_buy_action", "keep_cash")).strip().lower()
    if unfilled_buy_action != "keep_cash":
        raise ValueError("execution_sim.unfilled_buy_action must be 'keep_cash'.")
    unfilled_sell_action = str(sim_cfg.get("unfilled_sell_action", "keep_position")).strip().lower()
    if unfilled_sell_action != "keep_position":
        raise ValueError("execution_sim.unfilled_sell_action must be 'keep_position'.")

    round_lot_raw = sim_cfg.get("round_lot")
    if round_lot_raw is None:
        round_lot = None
    else:
        round_lot = _coerce_positive_int(round_lot_raw, label="execution_sim.round_lot")
    enforce_t1 = bool(sim_cfg.get("enforce_t1", False))
    enforce_price_limits = bool(sim_cfg.get("enforce_price_limits", False))
    enforce_listing_status = bool(sim_cfg.get("enforce_listing_status", False))
    limit_up_col = _coerce_optional_str(sim_cfg.get("limit_up_col"))
    limit_down_col = _coerce_optional_str(sim_cfg.get("limit_down_col"))
    listing_status_col = _coerce_optional_str(sim_cfg.get("listing_status_col"))
    lot_tolerance = float(cast("float", sim_cfg.get("lot_tolerance", 1e-6)))
    if not np.isfinite(lot_tolerance) or lot_tolerance < 0:
        raise ValueError("execution_sim.lot_tolerance must be >= 0.")

    return ExecutionSimConfig(
        enabled=True,
        portfolio_value=portfolio_value,
        participation_rate=participation_rate,
        liquidity_cols=liquidity_cols,
        buy_max_days=buy_max_days,
        sell_max_days=sell_max_days,
        zero_fill_abort_days_buy=zero_fill_abort_days_buy,
        unfilled_buy_action=unfilled_buy_action,
        unfilled_sell_action=unfilled_sell_action,
        round_lot=round_lot,
        enforce_t1=enforce_t1,
        enforce_price_limits=enforce_price_limits,
        enforce_listing_status=enforce_listing_status,
        limit_up_col=limit_up_col,
        limit_down_col=limit_down_col,
        listing_status_col=listing_status_col,
        lot_tolerance=lot_tolerance,
    )


def required_execution_sim_columns(
    config: ExecutionSimConfig,
    *,
    price_col: str,
    tradable_col: str | None,
) -> set[str]:
    del tradable_col
    if not config.enabled:
        return set()
    columns = {str(price_col), *config.liquidity_cols}
    return {col for col in columns if col}


def describe_execution_sim_config(config: ExecutionSimConfig) -> dict[str, Any]:
    return {
        "enabled": bool(config.enabled),
        "portfolio_value": float(config.portfolio_value),
        "participation_rate": float(config.participation_rate),
        "liquidity_cols": list(config.liquidity_cols),
        "buy_max_days": int(config.buy_max_days),
        "sell_max_days": config.sell_max_days,
        "zero_fill_abort_days_buy": config.zero_fill_abort_days_buy,
        "unfilled_buy_action": config.unfilled_buy_action,
        "unfilled_sell_action": config.unfilled_sell_action,
        "round_lot": config.round_lot,
        "enforce_t1": bool(config.enforce_t1),
        "enforce_price_limits": bool(config.enforce_price_limits),
        "enforce_listing_status": bool(config.enforce_listing_status),
        "limit_up_col": config.limit_up_col,
        "limit_down_col": config.limit_down_col,
        "listing_status_col": config.listing_status_col,
        "lot_tolerance": float(config.lot_tolerance),
    }


def _coerce_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _resolve_liquidity_cols(
    cfg: Mapping[str, Any],
    *,
    default_liquidity_col: str,
) -> tuple[str, ...]:
    raw_cols = cfg.get("liquidity_cols")
    if raw_cols is None:
        raw_cols = [cfg.get("liquidity_col", default_liquidity_col)]
    elif isinstance(raw_cols, str):
        raw_cols = [raw_cols]
    else:
        raw_cols = list(raw_cols)

    if bool(cfg.get("cap_daily_amount", True)):
        daily_col = str(cfg.get("daily_amount_col", "amount")).strip()
        if daily_col:
            raw_cols.append(daily_col)

    cols = [str(col).strip() for col in raw_cols if str(col).strip()]
    cols = list(dict.fromkeys(cols))
    if not cols:
        raise ValueError("execution_sim.liquidity_cols must not be empty.")
    return tuple(cols)


def _resolve_sell_max_days(value: object) -> int | str:
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {SELL_UNTIL_NEXT_REBALANCE, "until_next", "next_rebalance"}:
            return SELL_UNTIL_NEXT_REBALANCE
    return _coerce_positive_int(value, label="execution_sim.sell_max_days")


def _coerce_positive_float(value: Any, *, label: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise ValueError(f"{label} must be > 0.")
    return number


def _coerce_positive_int(value: Any, *, label: str) -> int:
    number = int(value)
    if number <= 0:
        raise ValueError(f"{label} must be a positive integer.")
    return number
