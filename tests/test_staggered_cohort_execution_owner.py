from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_backtester import (
    StaggeredCohortExecutionConfig,
    execution_summary_frame,
    simulate_staggered_cohort_execution,
    summarize_staggered_execution,
)
from portfolio_backtester.daily_watch20_policy import DailyWatch20PortfolioPolicy


def _prices(
    dates: list[str],
    symbols: tuple[str, ...] = ("AAA",),
    *,
    opens: dict[tuple[str, str], float] | None = None,
    suspended: set[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    opens = opens or {}
    suspended = suspended or set()
    rows: list[dict[str, object]] = []
    for date in dates:
        for symbol in symbols:
            price = opens.get((date, symbol), 10.0)
            rows.append(
                {
                    "trade_date": date,
                    "symbol": symbol,
                    "open": price,
                    "up_limit": price * 1.1,
                    "down_limit": price * 0.9,
                    "is_suspended": (date, symbol) in suspended,
                }
            )
    return pd.DataFrame(rows)


def _config(**overrides: object) -> StaggeredCohortExecutionConfig:
    values: dict[str, object] = {
        "horizon_days": 3,
        "top_n": 1,
        "initial_capital": 300.0,
        "single_side_cost_bps": 0.0,
    }
    values.update(overrides)
    return StaggeredCohortExecutionConfig(**values)  # type: ignore[arg-type]


def _calendar(dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"cal_date": dates, "is_open": [1] * len(dates)})


def _signals(dates: list[str], symbols: list[str], scores: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": dates,
            "available_at": [f"{date}T15:01:00+08:00" for date in dates],
            "symbol": symbols,
            "score": scores,
        }
    )


def test_h3_daily_signals_rotate_three_independent_cohorts() -> None:
    dates = ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07", "2025-01-08"]
    signals = _signals(dates[:3], ["AAA", "BBB", "CCC"], [3.0, 2.0, 1.0])
    result = simulate_staggered_cohort_execution(
        signals,
        _prices(dates, ("AAA", "BBB", "CCC")),
        _config(),
        trade_calendar=_calendar(dates),
    )
    buys = result.orders.loc[result.orders["side"].eq("buy")]
    assert buys["trade_date"].dt.strftime("%Y-%m-%d").tolist() == dates[1:4]
    assert buys["cohort_id"].tolist() == [0, 1, 2]
    assert buys["filled_notional"].tolist() == pytest.approx([100.0, 100.0, 100.0])
    jan7 = result.cohort_daily.loc[result.cohort_daily["trade_date"].eq("2025-01-07")]
    assert jan7["cohort_nav"].sum() == pytest.approx(result.daily.iloc[2]["net_nav"])


@pytest.mark.parametrize("block_type", ["limit_up", "suspended"])
def test_blocked_buy_keeps_fixed_slot_in_cash(block_type: str) -> None:
    dates = ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07", "2025-01-08"]
    pricing = _prices(dates)
    entry = pricing["trade_date"].eq("2025-01-03")
    if block_type == "limit_up":
        pricing.loc[entry, "up_limit"] = 10.0
    else:
        pricing.loc[entry, "is_suspended"] = True
    result = simulate_staggered_cohort_execution(
        _signals([dates[0]], ["AAA"], [1.0]),
        pricing,
        _config(),
        trade_calendar=_calendar(dates),
    )
    buy = result.orders.iloc[0]
    assert buy["status"] == "blocked"
    assert buy["blocked_reason"] == ("limit_up_open" if block_type == "limit_up" else "suspended")
    assert result.daily.iloc[0]["cash"] == pytest.approx(300.0)
    assert result.generations.iloc[0]["status"] == "cash_only"


@pytest.mark.parametrize("block_type", ["limit_down", "suspended"])
def test_blocked_sell_carries_until_first_tradable_open(block_type: str) -> None:
    dates = [
        "2025-01-02",
        "2025-01-03",
        "2025-01-06",
        "2025-01-07",
        "2025-01-08",
        "2025-01-09",
    ]
    opens = {
        ("2025-01-03", "AAA"): 10.0,
        ("2025-01-08", "AAA"): 9.0,
        ("2025-01-09", "AAA"): 8.0,
    }
    pricing = _prices(dates, opens=opens)
    planned_exit = pricing["trade_date"].eq("2025-01-08")
    if block_type == "limit_down":
        pricing.loc[planned_exit, "down_limit"] = 9.0
    else:
        pricing.loc[planned_exit, "is_suspended"] = True
    result = simulate_staggered_cohort_execution(
        _signals([dates[0]], ["AAA"], [1.0]),
        pricing,
        _config(),
        trade_calendar=_calendar(dates),
    )
    sells = result.orders.loc[result.orders["side"].eq("sell")]
    assert sells["status"].tolist() == ["blocked", "filled"]
    jan8 = result.positions.loc[result.positions["trade_date"].eq("2025-01-08")]
    assert jan8.iloc[0]["is_carry"]
    assert jan8.iloc[0]["carry_days"] == 1
    assert result.generations.iloc[0]["gross_return"] == pytest.approx(-0.2)
    assert result.summary["complete_nav"] is True


def test_cash_positions_and_costs_conserve_nav() -> None:
    dates = ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07", "2025-01-08"]
    signals = _signals([dates[0], dates[0]], ["AAA", "BBB"], [2.0, 1.0])
    result = simulate_staggered_cohort_execution(
        signals,
        _prices(dates, ("AAA", "BBB")),
        _config(top_n=2, single_side_cost_bps=10.0),
        trade_calendar=_calendar(dates),
    )
    assert np.allclose(
        result.daily["cash"] + result.daily["positions_value"], result.daily["net_nav"]
    )
    assert np.allclose(result.daily["cash_weight"] + result.daily["gross_exposure"], 1.0)
    fills = result.orders.loc[result.orders["status"].eq("filled")]
    assert fills["transaction_cost"].sum() == pytest.approx(
        fills["filled_notional"].sum() * 10.0 / 10_000.0
    )
    assert np.allclose(
        result.daily["net_nav"],
        result.daily["gross_nav_before_cost"] - result.daily["transaction_cost"],
    )


def test_adjusted_valuation_price_audits_raw_open() -> None:
    dates = ["2025-01-02", "2025-01-03", "2025-01-06"]
    pricing = _prices(
        dates,
        opens={(dates[1], "AAA"): 10.0, (dates[2], "AAA"): 5.0},
    )
    pricing["adj_open"] = pricing["open"]
    pricing.loc[pricing["trade_date"].isin(dates[1:]), "adj_open"] = 20.0
    result = simulate_staggered_cohort_execution(
        _signals([dates[0]], ["AAA"], [1.0]),
        pricing,
        _config(horizon_days=1, valuation_price_col="adj_open"),
        trade_calendar=_calendar(dates),
    )
    assert result.generations.iloc[0]["gross_return"] == pytest.approx(0.0)
    assert result.summary["tradability_price"] == "raw_open"
    orders = result.orders.set_index("side")
    assert orders.loc["buy", "raw_open"] == pytest.approx(10.0)
    assert orders.loc["buy", "valuation_open"] == pytest.approx(20.0)


def test_terminal_positions_fail_closed() -> None:
    dates = ["2025-01-02", "2025-01-03", "2025-01-06"]
    result = simulate_staggered_cohort_execution(
        _signals([dates[0]], ["AAA"], [1.0]),
        _prices(dates),
        _config(),
        trade_calendar=_calendar(dates),
    )
    assert result.summary["status"] == "incomplete_terminal_positions"
    assert result.summary["final_nav"] is None
    assert pd.isna(result.generations.iloc[0]["gross_return"])


def test_public_execution_summary_and_policy() -> None:
    dates = ["2025-01-02", "2025-01-03", "2025-01-06"]
    result = simulate_staggered_cohort_execution(
        _signals([dates[0]], ["AAA"], [1.0]),
        _prices(dates, opens={(dates[1], "AAA"): 10.0, (dates[2], "AAA"): 11.0}),
        _config(horizon_days=1),
        trade_calendar=_calendar(dates),
    )
    summary = summarize_staggered_execution(
        result, variant="A", horizon=1, single_side_cost_bps=0.0
    )
    frame = execution_summary_frame([summary])
    assert frame.iloc[0]["total_return"] == pytest.approx(0.1 / 3.0)
    assert frame.iloc[0]["terminal_complete"]
    assert DailyWatch20PortfolioPolicy().to_dict()["terminal_policy"] == "fail_closed"
