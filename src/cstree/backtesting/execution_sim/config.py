from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

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
        sim_cfg,
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
    }


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


def _coerce_positive_float(value: object, *, label: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise ValueError(f"{label} must be > 0.")
    return number


def _coerce_positive_int(value: object, *, label: str) -> int:
    number = int(value)
    if number <= 0:
        raise ValueError(f"{label} must be a positive integer.")
    return number
