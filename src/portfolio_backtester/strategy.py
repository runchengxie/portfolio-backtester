from __future__ import annotations

from typing import Any

import pandas as pd

from .contracts import GroupCap, StrategySpec
from .execution import ExecutionModel
from .portfolio_positions import build_positions_by_rebalance


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def strategy_from_config(config: dict[str, Any]) -> StrategySpec:
    strategy_cfg = _mapping(config.get("strategy"))
    backtest_cfg = _mapping(config.get("backtest"))
    if strategy_cfg:
        group_cfg = _mapping(strategy_cfg.get("group_cap"))
        group_cap = None
        if group_cfg.get("column") and group_cfg.get("max_names") is not None:
            group_cap = GroupCap(
                column=str(group_cfg["column"]),
                max_names=int(group_cfg["max_names"]),
            )
        return StrategySpec(
            name=str(strategy_cfg.get("name") or strategy_cfg.get("type") or "strategy"),
            type=str(strategy_cfg.get("type") or "topk_buffered_long_only"),
            score_col=str(strategy_cfg.get("score_col") or "signal_backtest"),
            top_k=int(strategy_cfg.get("top_k", backtest_cfg.get("top_k", 5))),
            buffer_exit=int(strategy_cfg.get("buffer_exit", backtest_cfg.get("buffer_exit", 0))),
            buffer_entry=int(strategy_cfg.get("buffer_entry", backtest_cfg.get("buffer_entry", 0))),
            weighting=str(strategy_cfg.get("weighting", backtest_cfg.get("weighting", "equal"))),
            long_only=bool(strategy_cfg.get("long_only", backtest_cfg.get("long_only", True))),
            short_k=(
                int(strategy_cfg["short_k"])
                if strategy_cfg.get("short_k") is not None
                else (
                    int(backtest_cfg["short_k"])
                    if backtest_cfg.get("short_k") is not None
                    else None
                )
            ),
            group_cap=group_cap,
            execution=_mapping(strategy_cfg.get("execution")) or None,
            source="explicit",
        )
    group_cap = None
    if backtest_cfg.get("group_col") and backtest_cfg.get("max_names_per_group") is not None:
        group_cap = GroupCap(
            column=str(backtest_cfg["group_col"]),
            max_names=int(backtest_cfg["max_names_per_group"]),
        )
    top_k = int(backtest_cfg.get("top_k", 5))
    return StrategySpec(
        name=str(backtest_cfg.get("strategy_name") or f"legacy_topk_k{top_k}"),
        type="topk_buffered_long_only"
        if bool(backtest_cfg.get("long_only", True))
        else "topk_buffered_long_short",
        score_col="signal_backtest",
        top_k=top_k,
        buffer_exit=int(backtest_cfg.get("buffer_exit", 0)),
        buffer_entry=int(backtest_cfg.get("buffer_entry", 0)),
        weighting=str(backtest_cfg.get("weighting", "equal")),
        long_only=bool(backtest_cfg.get("long_only", True)),
        short_k=int(backtest_cfg["short_k"]) if backtest_cfg.get("short_k") is not None else None,
        group_cap=group_cap,
        source="legacy_backtest_mapping",
    )


def construct_positions_from_strategy(
    signals: pd.DataFrame,
    *,
    strategy: StrategySpec,
    price_col: str,
    rebalance_dates: list[pd.Timestamp],
    shift_days: int,
    execution: ExecutionModel | None = None,
) -> pd.DataFrame:
    group_col = strategy.group_cap.column if strategy.group_cap else None
    max_names = strategy.group_cap.max_names if strategy.group_cap else None
    data = signals.copy()
    if "signal_date" in data.columns and "trade_date" not in data.columns:
        data["trade_date"] = pd.to_datetime(data["signal_date"], errors="coerce")
    if strategy.score_col not in data.columns:
        raise ValueError(f"Strategy score column not found: {strategy.score_col}")
    return build_positions_by_rebalance(
        data,
        pred_col=strategy.score_col,
        price_col=price_col,
        rebalance_dates=rebalance_dates,
        top_k=strategy.top_k,
        shift_days=shift_days,
        weighting=strategy.weighting,
        buffer_exit=strategy.buffer_exit,
        buffer_entry=strategy.buffer_entry,
        long_only=strategy.long_only,
        short_k=strategy.short_k,
        group_col=group_col,
        max_names_per_group=max_names,
        execution=execution,
    )


def strategy_lineage(
    strategy: StrategySpec,
    *,
    signals_file: str | None = None,
    positions_file: str | None = None,
) -> dict[str, Any]:
    return {
        **strategy.to_dict(),
        "signals_file": signals_file,
        "positions_file": positions_file,
    }
