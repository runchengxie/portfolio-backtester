"""Native deterministic position-replay backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal

import pandas as pd

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
    """

    positions: pd.DataFrame
    pricing: pd.DataFrame
    periods: pd.DataFrame
    config: PositionBacktestConfig
    intraday_bars: pd.DataFrame | None = None
    intraday_execution_assumption: IntradayExecutionAssumption | None = None
    allow_stale_execution_price: bool = False


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
        result = run_position_backtest(
            positions=request.positions,
            pricing=request.pricing,
            periods=request.periods,
            config=request.config,
            intraday_bars=request.intraday_bars,
        )
        performance = result.net_returns.merge(
            result.gross_returns,
            on="period_end",
            how="outer",
            validate="one_to_one",
        ).sort_values("period_end", kind="stable")
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

    if request.intraday_bars is not None and not request.intraday_bars.empty:
        if request.intraday_execution_assumption is None:
            raise ValueError(
                "Intraday VWAP replay requires intraday_execution_assumption to prevent "
                "accidental use of prices observed before the decision time."
            )


__all__ = [
    "IntradayExecutionAssumption",
    "NativePositionReplayBackend",
    "NativePositionReplayRequest",
]
