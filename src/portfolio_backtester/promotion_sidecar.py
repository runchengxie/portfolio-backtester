from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, cast

import pandas as pd


@dataclass(frozen=True)
class PromotionSidecarConfig:
    enabled: bool = False
    portfolio_value: float = 1_000_000.0
    participation_rate: float = 0.1
    price_col: str = "close"
    amount_col: str = "amount"
    enforce_t1_sell: bool = True
    signal_entry_delay_days: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _PromotionLedger:
    cash: float
    holdings: dict[str, dict[str, Any]]
    events: list[dict[str, Any]]
    orders: list[dict[str, Any]]
    fills: list[dict[str, Any]]
    states: list[dict[str, Any]]
    cash_rows: list[dict[str, Any]]
    violations: list[dict[str, Any]]


def build_promotion_sidecar_config(value: Any) -> PromotionSidecarConfig:
    if value is None or value is False:
        return PromotionSidecarConfig(enabled=False)
    if value is True:
        return PromotionSidecarConfig(enabled=True)
    if not isinstance(value, dict):
        raise ValueError("promotion_sidecar must be a mapping or boolean.")
    return PromotionSidecarConfig(
        enabled=bool(value.get("enabled", False)),
        portfolio_value=float(value.get("portfolio_value", 1_000_000.0)),
        participation_rate=float(value.get("participation_rate", 0.1)),
        price_col=str(value.get("price_col", "close")),
        amount_col=str(value.get("amount_col", "amount")),
        enforce_t1_sell=bool(value.get("enforce_t1_sell", True)),
        signal_entry_delay_days=int(value.get("signal_entry_delay_days", 1)),
    )


def promotion_sidecar_config_from_pipeline_config(
    config: Mapping[str, Any] | None,
) -> PromotionSidecarConfig:
    """Resolve the sidecar section from a pipeline configuration mapping."""
    pipeline_config = config if isinstance(config, Mapping) else {}
    value = pipeline_config.get("promotion_sidecar")
    promotion = pipeline_config.get("promotion")
    if value is None and isinstance(promotion, Mapping):
        value = promotion.get("event_sidecar")
    return build_promotion_sidecar_config(value)


def _empty_result(config: PromotionSidecarConfig) -> dict[str, Any]:
    empty = pd.DataFrame()
    return {
        "enabled": config.enabled,
        "summary": {
            "enabled": config.enabled,
            "status": "disabled" if not config.enabled else "not_run",
            "config": config.to_dict(),
        },
        "events": empty,
        "orders": empty,
        "fills": empty,
        "positions": empty,
        "cash": empty,
        "violations": empty,
    }


def simulate_promotion_sidecar(
    positions: pd.DataFrame | None,
    pricing: pd.DataFrame | None,
    config: PromotionSidecarConfig,
    *,
    buy_tradable_col: str = "is_buy_tradable",
    sell_tradable_col: str = "is_sell_tradable",
) -> dict[str, Any]:
    if not config.enabled:
        return _empty_result(config)
    if positions is None or positions.empty:
        result = _empty_result(config)
        result["enabled"] = True
        result["summary"]["enabled"] = True
        result["summary"]["status"] = "no_positions"
        return result
    if pricing is None or pricing.empty:
        raise ValueError("promotion sidecar requires pricing data when enabled.")

    pos = _prepare_positions(positions, config)
    price_lookup = _prepare_price_lookup(pricing)
    ledger = _PromotionLedger(
        cash=float(config.portfolio_value),
        holdings={},
        events=[],
        orders=[],
        fills=[],
        states=[],
        cash_rows=[],
        violations=[],
    )

    for entry_date, group in pos.groupby("entry_date", sort=True):
        entry_ts = cast("pd.Timestamp", pd.Timestamp(cast("pd.Timestamp", entry_date)))
        _process_entry_date(
            ledger,
            entry_ts,
            group,
            price_lookup=price_lookup,
            config=config,
            buy_tradable_col=buy_tradable_col,
            sell_tradable_col=sell_tradable_col,
        )

    return _build_result(ledger, config)


def _prepare_positions(
    positions: pd.DataFrame,
    config: PromotionSidecarConfig,
) -> pd.DataFrame:
    pos = positions.copy()
    pos["entry_date"] = pd.to_datetime(pos["entry_date"], errors="coerce").dt.normalize()
    if "rebalance_date" in pos.columns:
        pos["rebalance_date"] = pd.to_datetime(
            pos["rebalance_date"], errors="coerce"
        ).dt.normalize()
    else:
        pos["rebalance_date"] = pos["entry_date"] - pd.Timedelta(
            days=config.signal_entry_delay_days
        )
    return pos


def _prepare_price_lookup(pricing: pd.DataFrame) -> pd.DataFrame:
    price = pricing.copy()
    price["trade_date"] = pd.to_datetime(price["trade_date"], errors="coerce").dt.normalize()
    price = price.dropna(subset=["trade_date", "symbol"])
    return price.set_index(["trade_date", "symbol"])


def _format_entry_date(entry_date: pd.Timestamp) -> str:
    return entry_date.strftime("%Y%m%d")


def _price_row(
    price_lookup: pd.DataFrame,
    entry_date: pd.Timestamp,
    symbol: str,
) -> pd.Series | None:
    row_key = (pd.Timestamp(entry_date), symbol)
    return price_lookup.loc[row_key] if row_key in price_lookup.index else None


def _append_signal_and_target_events(
    ledger: _PromotionLedger,
    entry_date: pd.Timestamp,
    group: pd.DataFrame,
) -> None:
    for _, row in group.iterrows():
        symbol = str(row["symbol"])
        signal_date = row.get("rebalance_date")
        signal_ts = cast("pd.Timestamp", pd.Timestamp(signal_date))
        signal_date_str = signal_ts.strftime("%Y%m%d") if not pd.isna(signal_ts) else ""
        ledger.events.append(
            {
                "event_type": "SignalEvent",
                "signal_date": signal_date_str,
                "symbol": symbol,
                "target_weight": float(row.get("weight", 0.0)),
            }
        )
        ledger.events.append(
            {
                "event_type": "TargetEvent",
                "entry_date": _format_entry_date(entry_date),
                "symbol": symbol,
                "target_weight": float(row.get("weight", 0.0)),
            }
        )


def _append_violation(
    ledger: _PromotionLedger,
    entry_date: pd.Timestamp,
    *,
    symbol: str,
    reason: str | None,
) -> None:
    ledger.violations.append(
        {
            "entry_date": _format_entry_date(entry_date),
            "symbol": symbol,
            "constraint": reason,
        }
    )


def _append_order(
    ledger: _PromotionLedger,
    entry_date: pd.Timestamp,
    *,
    symbol: str,
    side: str,
    target_value: float,
    status: str,
    reason: str | None,
) -> None:
    ledger.orders.append(
        {
            "event_type": "OrderIntent",
            "entry_date": _format_entry_date(entry_date),
            "symbol": symbol,
            "side": side,
            "target_value": target_value,
            "status": status,
            "reason": reason,
        }
    )


def _append_fill(
    ledger: _PromotionLedger,
    entry_date: pd.Timestamp,
    *,
    symbol: str,
    side: str,
    filled: float,
    unfilled: float,
) -> None:
    ledger.fills.append(
        {
            "event_type": "FillEvent",
            "entry_date": _format_entry_date(entry_date),
            "symbol": symbol,
            "side": side,
            "filled_value": filled,
            "unfilled_value": unfilled,
        }
    )


def _sell_block_reason(
    ledger: _PromotionLedger,
    entry_date: pd.Timestamp,
    *,
    symbol: str,
    price_row: pd.Series | None,
    config: PromotionSidecarConfig,
    sell_tradable_col: str,
) -> str | None:
    if config.enforce_t1_sell and ledger.holdings[symbol].get("buy_date") == entry_date:
        return "t1_sell_block"
    if (
        price_row is not None
        and sell_tradable_col in price_row
        and not bool(price_row[sell_tradable_col])
    ):
        return "sell_not_tradable"
    return None


def _process_sell_orders(
    ledger: _PromotionLedger,
    entry_date: pd.Timestamp,
    target_symbols: set[str],
    *,
    price_lookup: pd.DataFrame,
    config: PromotionSidecarConfig,
    sell_tradable_col: str,
) -> None:
    for symbol in sorted(set(ledger.holdings) - target_symbols):
        price_row = _price_row(price_lookup, entry_date, symbol)
        reason = _sell_block_reason(
            ledger,
            entry_date,
            symbol=symbol,
            price_row=price_row,
            config=config,
            sell_tradable_col=sell_tradable_col,
        )
        target_value = float(ledger.holdings[symbol].get("market_value", 0.0))
        _append_order(
            ledger,
            entry_date,
            symbol=symbol,
            side="sell",
            target_value=target_value,
            status="blocked" if reason else "submitted",
            reason=reason,
        )
        if reason:
            _append_violation(ledger, entry_date, symbol=symbol, reason=reason)
            continue
        ledger.cash += target_value
        _append_fill(
            ledger,
            entry_date,
            symbol=symbol,
            side="sell",
            filled=target_value,
            unfilled=0.0,
        )
        del ledger.holdings[symbol]


def _buy_capacity_and_reason(
    price_row: pd.Series | None,
    *,
    target_value: float,
    config: PromotionSidecarConfig,
    buy_tradable_col: str,
) -> tuple[float, str | None]:
    if price_row is None:
        return 0.0, "missing_price"
    reason = None
    if buy_tradable_col in price_row and not bool(price_row[buy_tradable_col]):
        reason = "buy_not_tradable"
    capacity = target_value
    if config.amount_col in price_row and pd.notna(price_row[config.amount_col]):
        capacity = max(0.0, float(price_row[config.amount_col]) * float(config.participation_rate))
    return capacity, reason


def _process_buy_orders(
    ledger: _PromotionLedger,
    entry_date: pd.Timestamp,
    group: pd.DataFrame,
    *,
    price_lookup: pd.DataFrame,
    config: PromotionSidecarConfig,
    buy_tradable_col: str,
) -> None:
    for _, row in group.iterrows():
        symbol = str(row["symbol"])
        target_value = float(row.get("weight", 0.0)) * float(config.portfolio_value)
        price_row = _price_row(price_lookup, entry_date, symbol)
        capacity, reason = _buy_capacity_and_reason(
            price_row,
            target_value=target_value,
            config=config,
            buy_tradable_col=buy_tradable_col,
        )
        filled = 0.0 if reason else min(target_value, capacity, ledger.cash)
        unfilled = max(0.0, target_value - filled)
        status = "blocked" if reason else ("partial" if unfilled > 1e-8 else "filled")
        _append_order(
            ledger,
            entry_date,
            symbol=symbol,
            side="buy",
            target_value=target_value,
            status=status,
            reason=reason,
        )
        if reason:
            _append_violation(ledger, entry_date, symbol=symbol, reason=reason)
        ledger.cash -= filled
        _append_fill(
            ledger,
            entry_date,
            symbol=symbol,
            side="buy",
            filled=filled,
            unfilled=unfilled,
        )
        if filled > 0:
            ledger.holdings[symbol] = {
                "buy_date": entry_date,
                "market_value": filled,
                "weight": filled / float(config.portfolio_value),
            }


def _append_position_and_cash_state(
    ledger: _PromotionLedger,
    entry_date: pd.Timestamp,
    *,
    config: PromotionSidecarConfig,
) -> None:
    for symbol, state in sorted(ledger.holdings.items()):
        ledger.states.append(
            {
                "event_type": "PositionState",
                "entry_date": _format_entry_date(entry_date),
                "symbol": symbol,
                "market_value": float(state["market_value"]),
                "weight": float(state["weight"]),
            }
        )
    ledger.cash_rows.append(
        {
            "entry_date": _format_entry_date(entry_date),
            "cash": ledger.cash,
            "cash_weight": ledger.cash / float(config.portfolio_value),
        }
    )


def _process_entry_date(
    ledger: _PromotionLedger,
    entry_date: pd.Timestamp,
    group: pd.DataFrame,
    *,
    price_lookup: pd.DataFrame,
    config: PromotionSidecarConfig,
    buy_tradable_col: str,
    sell_tradable_col: str,
) -> None:
    target_symbols = set(group["symbol"].astype(str))
    _append_signal_and_target_events(ledger, entry_date, group)
    _process_sell_orders(
        ledger,
        entry_date,
        target_symbols,
        price_lookup=price_lookup,
        config=config,
        sell_tradable_col=sell_tradable_col,
    )
    _process_buy_orders(
        ledger,
        entry_date,
        group,
        price_lookup=price_lookup,
        config=config,
        buy_tradable_col=buy_tradable_col,
    )
    _append_position_and_cash_state(ledger, entry_date, config=config)


def _build_summary(
    ledger: _PromotionLedger,
    config: PromotionSidecarConfig,
    *,
    fills_df: pd.DataFrame,
    orders_df: pd.DataFrame,
) -> dict[str, Any]:
    total_requested = float(pd.to_numeric(orders_df.get("target_value"), errors="coerce").sum())
    total_filled = float(pd.to_numeric(fills_df.get("filled_value"), errors="coerce").sum())
    delayed_sells = int((orders_df.get("status") == "blocked").sum()) if not orders_df.empty else 0
    return {
        "enabled": True,
        "status": "ok",
        "config": config.to_dict(),
        "order_count": len(ledger.orders),
        "fill_count": len(ledger.fills),
        "fill_ratio": total_filled / total_requested if total_requested > 0 else 0.0,
        "unfilled_notional": float(
            pd.to_numeric(fills_df.get("unfilled_value"), errors="coerce").sum()
        )
        if not fills_df.empty
        else 0.0,
        "delayed_sell_count": delayed_sells,
        "constraint_count": len(ledger.violations),
        "cash_end": float(ledger.cash),
        "cash_drag": float(ledger.cash / config.portfolio_value),
    }


def _build_result(ledger: _PromotionLedger, config: PromotionSidecarConfig) -> dict[str, Any]:
    orders_df = pd.DataFrame(ledger.orders)
    fills_df = pd.DataFrame(ledger.fills)
    return {
        "enabled": True,
        "summary": _build_summary(ledger, config, fills_df=fills_df, orders_df=orders_df),
        "events": pd.DataFrame(ledger.events),
        "orders": orders_df,
        "fills": fills_df,
        "positions": pd.DataFrame(ledger.states),
        "cash": pd.DataFrame(ledger.cash_rows),
        "violations": pd.DataFrame(ledger.violations),
    }
