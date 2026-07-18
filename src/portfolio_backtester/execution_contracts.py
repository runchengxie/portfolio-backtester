"""Framework-neutral execution and accounting contracts.

The contracts in this module deliberately avoid importing Backtrader, vn.py,
Qlib, or any broker SDK. Adapters may translate framework objects into these
stable types, but framework-specific objects must not cross repository
boundaries or become persisted artifacts.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import StrEnum
from math import isfinite
from typing import Any, Literal


class OrderStatus(StrEnum):
    """Canonical order lifecycle states shared by replay and transport adapters."""

    CREATED = "created"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    REJECTED = "rejected"


class OrderType(StrEnum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class TimeInForce(StrEnum):
    DAY = "day"
    GOOD_TIL_CANCELLED = "gtc"
    IMMEDIATE_OR_CANCEL = "ioc"
    FILL_OR_KILL = "fok"


TERMINAL_ORDER_STATUSES = frozenset(
    {
        OrderStatus.FILLED,
        OrderStatus.CANCELLED,
        OrderStatus.EXPIRED,
        OrderStatus.REJECTED,
    }
)

_ALLOWED_ORDER_TRANSITIONS: Mapping[OrderStatus, frozenset[OrderStatus]] = {
    OrderStatus.CREATED: frozenset(
        {OrderStatus.SUBMITTED, OrderStatus.CANCELLED, OrderStatus.REJECTED}
    ),
    OrderStatus.SUBMITTED: frozenset(
        {
            OrderStatus.ACCEPTED,
            OrderStatus.PARTIAL,
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.EXPIRED,
            OrderStatus.REJECTED,
        }
    ),
    OrderStatus.ACCEPTED: frozenset(
        {
            OrderStatus.PARTIAL,
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.EXPIRED,
            OrderStatus.REJECTED,
        }
    ),
    OrderStatus.PARTIAL: frozenset(
        {
            OrderStatus.PARTIAL,
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.EXPIRED,
        }
    ),
    OrderStatus.FILLED: frozenset(),
    OrderStatus.CANCELLED: frozenset(),
    OrderStatus.EXPIRED: frozenset(),
    OrderStatus.REJECTED: frozenset(),
}


def _assert_timezone_aware(value: datetime, *, label: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware.")


def assert_order_transition(current: OrderStatus, next_status: OrderStatus) -> None:
    """Reject impossible lifecycle transitions while allowing exact replays."""

    if current == next_status:
        return
    if next_status not in _ALLOWED_ORDER_TRANSITIONS[current]:
        raise ValueError(f"Invalid order transition: {current.value} -> {next_status.value}")


@dataclass(frozen=True)
class Instrument:
    """Minimal instrument metadata needed by research execution simulators."""

    symbol: str
    exchange: str = ""
    lot_size: int = 1
    price_tick: float = 0.01
    settlement_cycle: str = "T+0"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.symbol.strip():
            raise ValueError("Instrument.symbol must be non-empty.")
        if self.lot_size <= 0:
            raise ValueError("Instrument.lot_size must be a positive integer.")
        if not isfinite(self.price_tick) or self.price_tick <= 0:
            raise ValueError("Instrument.price_tick must be finite and > 0.")
        if not self.settlement_cycle.strip():
            raise ValueError("Instrument.settlement_cycle must be non-empty.")

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["metadata"] = dict(self.metadata)
        return record


@dataclass(frozen=True)
class Target:
    """A framework-neutral target produced by portfolio construction."""

    target_id: str
    symbol: str
    weight: float
    side: Literal["long", "short"]
    rebalance_time: datetime
    decision_time: datetime
    valid_until: datetime | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.target_id.strip():
            raise ValueError("Target.target_id must be non-empty.")
        if not self.symbol.strip():
            raise ValueError("Target.symbol must be non-empty.")
        if not isfinite(self.weight) or self.weight < 0:
            raise ValueError("Target.weight must be finite and >= 0.")
        _assert_timezone_aware(self.decision_time, label="Target.decision_time")
        _assert_timezone_aware(self.rebalance_time, label="Target.rebalance_time")
        if self.valid_until is not None:
            _assert_timezone_aware(self.valid_until, label="Target.valid_until")
        if self.decision_time > self.rebalance_time:
            raise ValueError("Target.decision_time must not be after rebalance_time.")
        if self.valid_until is not None and self.valid_until < self.rebalance_time:
            raise ValueError("Target.valid_until must not be before rebalance_time.")

    def to_record(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "symbol": self.symbol,
            "weight": float(self.weight),
            "side": self.side,
            "rebalance_time": self.rebalance_time.isoformat(),
            "decision_time": self.decision_time.isoformat(),
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class OrderIntent:
    """Order request before a broker or simulator accepts it."""

    order_id: str
    target_id: str
    symbol: str
    side: Literal["buy", "sell"]
    quantity: float
    submitted_at: datetime
    order_type: OrderType = OrderType.MARKET
    time_in_force: TimeInForce = TimeInForce.DAY
    limit_price: float | None = None
    stop_price: float | None = None
    replaces_order_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.order_id.strip():
            raise ValueError("OrderIntent.order_id must be non-empty.")
        if not self.target_id.strip():
            raise ValueError("OrderIntent.target_id must be non-empty.")
        if not self.symbol.strip():
            raise ValueError("OrderIntent.symbol must be non-empty.")
        if not isfinite(self.quantity) or self.quantity <= 0:
            raise ValueError("OrderIntent.quantity must be finite and > 0.")
        _assert_timezone_aware(self.submitted_at, label="OrderIntent.submitted_at")
        if self.order_type in {OrderType.LIMIT, OrderType.STOP_LIMIT} and (
            self.limit_price is None or not isfinite(self.limit_price) or self.limit_price <= 0
        ):
            raise ValueError("Limit orders require limit_price > 0.")
        if self.order_type in {OrderType.STOP, OrderType.STOP_LIMIT} and (
            self.stop_price is None or not isfinite(self.stop_price) or self.stop_price <= 0
        ):
            raise ValueError("Stop orders require stop_price > 0.")

    def to_record(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id,
            "target_id": self.target_id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": float(self.quantity),
            "submitted_at": self.submitted_at.isoformat(),
            "order_type": self.order_type.value,
            "time_in_force": self.time_in_force.value,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "replaces_order_id": self.replaces_order_id,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class OrderEvent:
    """Idempotent order status event suitable for replay and reconciliation."""

    event_id: str
    order_id: str
    status: OrderStatus
    event_time: datetime
    cumulative_quantity: float
    remaining_quantity: float
    reason: str | None = None

    def __post_init__(self) -> None:
        if not self.event_id.strip():
            raise ValueError("OrderEvent.event_id must be non-empty.")
        if not self.order_id.strip():
            raise ValueError("OrderEvent.order_id must be non-empty.")
        _assert_timezone_aware(self.event_time, label="OrderEvent.event_time")
        for label, value in (
            ("cumulative_quantity", self.cumulative_quantity),
            ("remaining_quantity", self.remaining_quantity),
        ):
            if not isfinite(value) or value < 0:
                raise ValueError(f"OrderEvent.{label} must be finite and >= 0.")
        if self.status == OrderStatus.FILLED and self.remaining_quantity > 1e-12:
            raise ValueError("Filled order events must have zero remaining_quantity.")

    def to_record(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "order_id": self.order_id,
            "status": self.status.value,
            "event_time": self.event_time.isoformat(),
            "cumulative_quantity": float(self.cumulative_quantity),
            "remaining_quantity": float(self.remaining_quantity),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class OrderState:
    order_id: str
    status: OrderStatus
    cumulative_quantity: float
    remaining_quantity: float
    last_event_time: datetime
    applied_event_ids: tuple[str, ...]


@dataclass(frozen=True)
class Fill:
    """One execution of an order; one order may produce many fills."""

    fill_id: str
    order_id: str
    symbol: str
    side: Literal["buy", "sell"]
    quantity: float
    price: float
    filled_at: datetime
    fee: float = 0.0
    slippage: float = 0.0

    def __post_init__(self) -> None:
        for label, value in (("quantity", self.quantity), ("price", self.price)):
            if not isfinite(value) or value <= 0:
                raise ValueError(f"Fill.{label} must be finite and > 0.")
        for label, value in (("fee", self.fee), ("slippage", self.slippage)):
            if not isfinite(value) or value < 0:
                raise ValueError(f"Fill.{label} must be finite and >= 0.")
        if not self.fill_id.strip() or not self.order_id.strip() or not self.symbol.strip():
            raise ValueError("Fill identifiers and symbol must be non-empty.")
        _assert_timezone_aware(self.filled_at, label="Fill.filled_at")

    def to_record(self) -> dict[str, Any]:
        return {
            "fill_id": self.fill_id,
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": float(self.quantity),
            "price": float(self.price),
            "filled_at": self.filled_at.isoformat(),
            "fee": float(self.fee),
            "slippage": float(self.slippage),
        }


@dataclass(frozen=True)
class LedgerSnapshot:
    """Cash and marked positions at one valuation time."""

    as_of: datetime
    cash: float
    positions_value: float
    nav: float
    gross_exposure: float
    accrued_fees: float = 0.0

    def __post_init__(self) -> None:
        _assert_timezone_aware(self.as_of, label="LedgerSnapshot.as_of")
        for label, value in (
            ("cash", self.cash),
            ("positions_value", self.positions_value),
            ("nav", self.nav),
            ("gross_exposure", self.gross_exposure),
            ("accrued_fees", self.accrued_fees),
        ):
            if not isfinite(value):
                raise ValueError(f"LedgerSnapshot.{label} must be finite.")
        if self.accrued_fees < 0:
            raise ValueError("LedgerSnapshot.accrued_fees must be >= 0.")
        self.assert_balanced()

    def assert_balanced(self, *, tolerance: float = 1e-8) -> None:
        expected_nav = self.cash + self.positions_value
        scale = max(1.0, abs(self.nav), abs(expected_nav))
        if abs(self.nav - expected_nav) > tolerance * scale:
            raise ValueError(
                "LedgerSnapshot must satisfy nav = cash + positions_value; "
                f"got nav={self.nav}, expected={expected_nav}."
            )

    def to_record(self) -> dict[str, Any]:
        return {
            "as_of": self.as_of.isoformat(),
            "cash": float(self.cash),
            "positions_value": float(self.positions_value),
            "nav": float(self.nav),
            "gross_exposure": float(self.gross_exposure),
            "accrued_fees": float(self.accrued_fees),
        }


def reduce_order_events(events: Iterable[OrderEvent]) -> OrderState:
    """Reduce duplicate and out-of-order callbacks into deterministic state.

    ``event_id`` is the idempotency key. Re-delivery of the same event is ignored;
    conflicting payloads for the same id are rejected. Events are then ordered by
    timestamp and id before lifecycle and quantity monotonicity checks are applied.
    """

    unique: dict[str, OrderEvent] = {}
    for event in events:
        existing = unique.get(event.event_id)
        if existing is not None and existing != event:
            raise ValueError(f"Conflicting payloads for event_id={event.event_id!r}.")
        unique[event.event_id] = event
    if not unique:
        raise ValueError("At least one order event is required.")

    ordered = sorted(unique.values(), key=lambda item: (item.event_time, item.event_id))
    order_ids = {event.order_id for event in ordered}
    if len(order_ids) != 1:
        raise ValueError("All reduced events must belong to the same order_id.")

    previous: OrderEvent | None = None
    for event in ordered:
        if previous is not None:
            assert_order_transition(previous.status, event.status)
            if event.cumulative_quantity + 1e-12 < previous.cumulative_quantity:
                raise ValueError("Order cumulative_quantity must be non-decreasing.")
            if event.remaining_quantity > previous.remaining_quantity + 1e-12:
                raise ValueError("Order remaining_quantity must be non-increasing.")
        previous = event

    final = ordered[-1]
    return OrderState(
        order_id=final.order_id,
        status=final.status,
        cumulative_quantity=float(final.cumulative_quantity),
        remaining_quantity=float(final.remaining_quantity),
        last_event_time=final.event_time,
        applied_event_ids=tuple(event.event_id for event in ordered),
    )


__all__ = [
    "TERMINAL_ORDER_STATUSES",
    "Fill",
    "Instrument",
    "LedgerSnapshot",
    "OrderEvent",
    "OrderIntent",
    "OrderState",
    "OrderStatus",
    "OrderType",
    "Target",
    "TimeInForce",
    "assert_order_transition",
    "reduce_order_events",
]
