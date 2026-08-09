"""Native deterministic position-replay backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal

import pandas as pd

from .._position_backtest_config import PositionBacktestResult
from ..execution_sim import ExecutionSimConfig, simulate_execution_adjusted_nav
from ..position_backtest import PositionBacktestConfig, run_position_backtest
from .base import (
    BackendCapabilities,
    CanonicalBacktestResult,
    to_json_compatible,
)

IntradayExecutionAssumption = Literal["signal_before_session", "caller_windowed"]


@dataclass(frozen=True)
class NativePositionReplayRequest:
    """Inputs for canonicalizing the existing period-return replay.

    The extra assumption fields deliberately make ambiguous historical behavior
    explicit. The compatibility API remains available, while this backend fails
    closed when a caller requests unsupported short positions, stale execution
    prices, or an unqualified full-session VWAP.

    ``ledger`` is an opt-in switch and does not change the default contract. When
    enabled, the period-return replay also runs the shared execution-sim engine
    and attaches orders/fills/daily_ledger. The default ``ledger=False`` keeps the
    historical ``not_available`` declaration so fixed-difference tests are stable.
    """

    positions: pd.DataFrame
    pricing: pd.DataFrame
    periods: pd.DataFrame
    config: PositionBacktestConfig
    intraday_bars: pd.DataFrame | None = None
    intraday_execution_assumption: IntradayExecutionAssumption | None = None
    allow_stale_execution_price: bool = False
    ledger: bool = False
    ledger_config: ExecutionSimConfig | None = None


class NativePositionReplayBackend:
    name: ClassVar[str] = "native.position_replay"
    capabilities: ClassVar[BackendCapabilities] = BackendCapabilities(
        target_generation=False,
        order_lifecycle=False,
        partial_fills=False,
        daily_ledger=False,
        long_short=False,
        market_rules=("tradability", "delayed_exit", "period_costs"),
    )

    def run(self, request: NativePositionReplayRequest) -> CanonicalBacktestResult:
        _validate_request(request)
        result: PositionBacktestResult = run_position_backtest(
            positions=request.positions,
            pricing=request.pricing,
            periods=request.periods,
            config=request.config,
            intraday_bars=request.intraday_bars,
        )
        performance = _aligned_performance_frame(
            result.net_returns,
            result.gross_returns,
        )
        if request.ledger:
            return self._run_with_ledger(request, result, performance)
        canonical = CanonicalBacktestResult(
            backend_name=self.name,
            capabilities=self.capabilities,
            performance=performance.reset_index(drop=True),
            positions=request.positions.copy(),
            summary=to_json_compatible(result.summary),
            metadata={
                "accounting_mode": "period_return_replay",
                "orders_and_fills": "not_available",
                "daily_ledger": "not_available",
                "intraday_execution_assumption": request.intraday_execution_assumption,
                "stale_execution_price_allowed": bool(request.allow_stale_execution_price),
                "native_result_schema": result.summary.get("schema"),
            },
        )
        canonical.validate()
        return canonical

    def _run_with_ledger(
        self,
        request: NativePositionReplayRequest,
        result: PositionBacktestResult,
        performance: pd.DataFrame,
    ) -> CanonicalBacktestResult:
        from ..position_backtest import normalize_position_backtest_positions

        sim_positions = normalize_position_backtest_positions(request.positions)
        sim_config = request.ledger_config or ExecutionSimConfig(enabled=True)
        price_col = request.config.price_col
        tradable_col = request.config.tradable_col
        ledger_result = simulate_execution_adjusted_nav(
            sim_positions,
            request.pricing,
            sim_config,
            price_col=price_col,
            tradable_col=tradable_col if tradable_col in request.pricing.columns else None,
            transaction_cost_bps=float(getattr(request.config, "transaction_cost_bps", 0.0) or 0.0),
            trading_days_per_year=int(getattr(request.config, "trading_days_per_year", 252) or 252),
        )
        ledger = ledger_result.to_unified_ledger(portfolio_value=float(sim_config.portfolio_value))
        orders = _attach_order_ids(ledger.orders)
        fills = _attach_fill_ids(ledger.fills, orders)
        daily_ledger = pd.DataFrame(
            {
                "trade_date": ledger.daily_cash["trade_date"].to_numpy(),
                "cash": ledger.daily_cash["cash"].to_numpy(),
                "positions_value": ledger.daily_positions["positions_value"].to_numpy(),
                "nav": ledger.daily_nav["nav"].to_numpy(),
            }
        )
        capabilities = BackendCapabilities(
            target_generation=False,
            order_lifecycle=True,
            partial_fills=True,
            daily_ledger=True,
            long_short=False,
            market_rules=("tradability", "delayed_exit", "period_costs"),
        )
        canonical = CanonicalBacktestResult(
            backend_name=self.name,
            capabilities=capabilities,
            performance=performance.reset_index(drop=True),
            positions=request.positions.copy(),
            orders=orders,
            fills=fills,
            daily_ledger=daily_ledger,
            summary=to_json_compatible(result.summary),
            metadata={
                "accounting_mode": "period_return_replay_with_ledger",
                "orders_and_fills": "execution_sim",
                "daily_ledger": "execution_sim",
                "intraday_execution_assumption": request.intraday_execution_assumption,
                "stale_execution_price_allowed": bool(request.allow_stale_execution_price),
                "native_result_schema": result.summary.get("schema")
                if hasattr(result, "summary")
                else None,
                "execution_sim_summary": to_json_compatible(ledger_result.summary),
            },
        )
        canonical.validate()
        return canonical


def _attach_order_ids(orders: pd.DataFrame) -> pd.DataFrame:
    if orders.empty:
        return orders.copy()
    out = orders.copy()
    key_cols = [c for c in ("rebalance_date", "entry_date", "symbol", "side") if c in out.columns]
    out["order_id"] = out.apply(lambda row: "|".join(str(row[c]) for c in key_cols), axis=1)
    return out


def _attach_fill_ids(fills: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    if fills.empty:
        return fills.copy()
    out = fills.copy()
    key_cols = [c for c in ("rebalance_date", "entry_date", "symbol", "side") if c in out.columns]
    out["order_id"] = out.apply(lambda row: "|".join(str(row[c]) for c in key_cols), axis=1)
    out["fill_id"] = out.apply(
        lambda row: "|".join(str(row[c]) for c in (*key_cols, "trade_date", "day_number")),
        axis=1,
    )
    return out


def _aligned_performance_frame(
    net_returns: pd.DataFrame,
    gross_returns: pd.DataFrame,
) -> pd.DataFrame:
    net = net_returns.reset_index(drop=True).copy()
    gross = gross_returns.reset_index(drop=True).copy()
    if net.shape[0] != gross.shape[0]:
        raise ValueError("Native net and gross return frames must have the same row count.")
    if not net["period_end"].astype(str).equals(gross["period_end"].astype(str)):
        raise ValueError("Native net and gross return frames must have aligned period_end rows.")
    net["gross_return"] = pd.to_numeric(gross["gross_return"], errors="coerce").to_numpy()
    return net


def _validate_request(request: NativePositionReplayRequest) -> None:
    if not request.config.long_only:
        raise ValueError(
            "NativePositionReplayBackend currently supports long-only positions; "
            "long_only=False would be silently narrowed by the compatibility replay."
        )

    if "side" in request.positions.columns:
        sides = request.positions["side"].astype(str).str.strip().str.lower()
        unsupported = sorted(set(sides.loc[~sides.eq("long")]))
        if unsupported:
            raise ValueError(
                "NativePositionReplayBackend does not support position side values: "
                + ", ".join(unsupported)
            )

    if "weight" in request.positions.columns:
        weights = pd.to_numeric(request.positions["weight"], errors="coerce")
        if weights.isna().any():
            raise ValueError("Native position weights must be numeric.")
        if (weights < 0).any():
            raise ValueError("NativePositionReplayBackend does not support negative weights.")

    if request.config.exit_price_policy == "ffill" and not request.allow_stale_execution_price:
        raise ValueError(
            "exit_price_policy='ffill' can mix valuation fallback with execution semantics. "
            "Set allow_stale_execution_price=True only for an explicitly documented study."
        )

    if (
        request.intraday_bars is not None
        and not request.intraday_bars.empty
        and request.intraday_execution_assumption is None
    ):
        raise ValueError(
            "Intraday VWAP replay requires intraday_execution_assumption to prevent "
            "accidental use of prices observed before the decision time."
        )


__all__ = [
    "IntradayExecutionAssumption",
    "NativePositionReplayBackend",
    "NativePositionReplayRequest",
]
