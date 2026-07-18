from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from portfolio_backtester.execution_contracts import (
    Fill,
    Instrument,
    LedgerSnapshot,
    OrderEvent,
    OrderIntent,
    OrderStatus,
    OrderType,
    Target,
    assert_order_transition,
    reduce_order_events,
)

T0 = datetime(2026, 1, 5, 9, 30, tzinfo=UTC)


def test_execution_contracts_validate_market_metadata_and_timestamps() -> None:
    instrument = Instrument(
        symbol="000001.SZ",
        exchange="XSHE",
        lot_size=100,
        price_tick=0.01,
        settlement_cycle="T+1",
    )
    target = Target(
        target_id="target-1",
        symbol=instrument.symbol,
        weight=0.25,
        side="long",
        decision_time=T0 - timedelta(hours=12),
        rebalance_time=T0,
    )

    assert instrument.to_record()["lot_size"] == 100
    assert target.to_record()["decision_time"].endswith("+00:00")


def test_target_rejects_decision_after_rebalance() -> None:
    with pytest.raises(ValueError, match="decision_time"):
        Target(
            target_id="target-1",
            symbol="AAA",
            weight=1.0,
            side="long",
            decision_time=T0 + timedelta(seconds=1),
            rebalance_time=T0,
        )


def test_execution_timestamps_must_be_timezone_aware() -> None:
    naive = datetime(2026, 1, 5, 9, 30)

    with pytest.raises(ValueError, match="timezone-aware"):
        OrderEvent(
            event_id="event-1",
            order_id="order-1",
            status=OrderStatus.SUBMITTED,
            event_time=naive,
            cumulative_quantity=0,
            remaining_quantity=100,
        )


def test_order_intent_requires_prices_for_conditional_orders() -> None:
    with pytest.raises(ValueError, match="limit_price"):
        OrderIntent(
            order_id="order-1",
            target_id="target-1",
            symbol="AAA",
            side="buy",
            quantity=100,
            submitted_at=T0,
            order_type=OrderType.LIMIT,
        )


def test_order_state_reduction_is_duplicate_and_order_insensitive() -> None:
    submitted = OrderEvent(
        event_id="event-1",
        order_id="order-1",
        status=OrderStatus.SUBMITTED,
        event_time=T0,
        cumulative_quantity=0,
        remaining_quantity=100,
    )
    partial = OrderEvent(
        event_id="event-2",
        order_id="order-1",
        status=OrderStatus.PARTIAL,
        event_time=T0 + timedelta(minutes=1),
        cumulative_quantity=40,
        remaining_quantity=60,
    )
    filled = OrderEvent(
        event_id="event-3",
        order_id="order-1",
        status=OrderStatus.FILLED,
        event_time=T0 + timedelta(minutes=2),
        cumulative_quantity=100,
        remaining_quantity=0,
    )

    state = reduce_order_events([filled, partial, submitted, partial])

    assert state.status == OrderStatus.FILLED
    assert state.cumulative_quantity == pytest.approx(100)
    assert state.remaining_quantity == pytest.approx(0)
    assert state.applied_event_ids == ("event-1", "event-2", "event-3")


def test_order_state_reduction_rejects_conflicting_duplicate() -> None:
    first = OrderEvent(
        event_id="event-1",
        order_id="order-1",
        status=OrderStatus.SUBMITTED,
        event_time=T0,
        cumulative_quantity=0,
        remaining_quantity=100,
    )
    conflicting = OrderEvent(
        event_id="event-1",
        order_id="order-1",
        status=OrderStatus.REJECTED,
        event_time=T0,
        cumulative_quantity=0,
        remaining_quantity=100,
    )

    with pytest.raises(ValueError, match="Conflicting payloads"):
        reduce_order_events([first, conflicting])


def test_terminal_order_transition_is_rejected() -> None:
    with pytest.raises(ValueError, match="filled -> submitted"):
        assert_order_transition(OrderStatus.FILLED, OrderStatus.SUBMITTED)


def test_fill_and_ledger_snapshot_enforce_accounting_invariants() -> None:
    fill = Fill(
        fill_id="fill-1",
        order_id="order-1",
        symbol="AAA",
        side="buy",
        quantity=100,
        price=10,
        filled_at=T0,
        fee=5,
    )
    snapshot = LedgerSnapshot(
        as_of=T0,
        cash=500,
        positions_value=500,
        nav=1000,
        gross_exposure=0.5,
        accrued_fees=5,
    )

    assert fill.to_record()["fee"] == pytest.approx(5)
    assert snapshot.to_record()["nav"] == pytest.approx(1000)

    with pytest.raises(ValueError, match="nav = cash"):
        LedgerSnapshot(
            as_of=T0,
            cash=500,
            positions_value=500,
            nav=999,
            gross_exposure=0.5,
        )
