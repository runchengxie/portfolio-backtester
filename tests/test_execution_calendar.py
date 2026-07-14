import numpy as np
import pandas as pd

from portfolio_backtester.engine import backtest_topk
from portfolio_backtester.evaluation import _filter_positions_to_backtest_periods
from portfolio_backtester.execution import build_execution_model
from portfolio_backtester.execution_calendar import (
    is_execution_open,
    resolve_execution_date,
    resolve_execution_open_dates,
)
from portfolio_backtester.portfolio import build_positions_by_rebalance


class FakeConnectCalendarRQ:
    def get_trading_dates(self, start_date, end_date, market="hk"):
        dates = pd.date_range(start_date, end_date, freq="D")
        if market == "hk":
            allowed = {"20260504", "20260505", "20260506"}
        elif market == "cn":  # RQData provider calendar alias for A-share sessions.
            allowed = {"20260506"}
        else:
            raise ValueError(f"unsupported market={market}")
        return [date for date in dates if date.strftime("%Y%m%d") in allowed]


def test_hk_connect_calendar_uses_hk_and_mainland_intersection() -> None:
    rq = FakeConnectCalendarRQ()

    open_dates = resolve_execution_open_dates(
        pd.Timestamp("2026-05-04"),
        pd.Timestamp("2026-05-06"),
        calendar="hk_connect",
        rqdatac_module=rq,
    )

    assert open_dates == [pd.Timestamp("2026-05-06")]
    assert not is_execution_open("2026-05-04", calendar="hk_connect", rqdatac_module=rq)
    assert is_execution_open("2026-05-06", calendar="hk_connect", rqdatac_module=rq)


def test_hk_connect_entry_date_can_resolve_future_open_day() -> None:
    rq = FakeConnectCalendarRQ()

    entry_date = resolve_execution_date(
        "2026-05-05",
        1,
        [pd.Timestamp("2026-05-04"), pd.Timestamp("2026-05-05")],
        calendar="hk_connect",
        rqdatac_module=rq,
        allow_future=True,
    )

    assert entry_date == pd.Timestamp("2026-05-06")


def test_live_position_builder_allows_explicit_future_entry_date() -> None:
    data = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2026-05-05", "2026-05-05"]),
            "symbol": ["00005.HK", "00700.HK"],
            "pred": [2.0, 1.0],
            "close": [50.0, 300.0],
        }
    )

    positions = build_positions_by_rebalance(
        data,
        pred_col="pred",
        price_col="close",
        rebalance_dates=[pd.Timestamp("2026-05-05")],
        top_k=1,
        shift_days=1,
        entry_dates_by_rebalance={pd.Timestamp("2026-05-05"): pd.Timestamp("2026-05-06")},
    )

    assert positions["rebalance_date"].tolist() == ["20260505"]
    assert positions["entry_date"].tolist() == ["20260506"]
    assert positions["symbol"].tolist() == ["00005.HK"]


def test_position_builder_uses_pricing_data_for_entry_calendar_and_selection() -> None:
    signal_data = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2020-01-31", "2020-01-31", "2020-02-28", "2020-02-28"]),
            "symbol": ["A", "B", "A", "B"],
            "pred": [10.0, 9.0, 10.0, 9.0],
        }
    )
    pricing_data = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(
                [
                    "2020-01-31",
                    "2020-01-31",
                    "2020-02-03",
                    "2020-02-03",
                    "2020-02-28",
                    "2020-02-28",
                    "2020-03-02",
                    "2020-03-02",
                ]
            ),
            "symbol": ["A", "B"] * 4,
            "close": [100.0, 100.0, np.nan, 100.0, 100.0, 100.0, 101.0, 100.0],
        }
    )

    positions = build_positions_by_rebalance(
        signal_data,
        pred_col="pred",
        price_col="close",
        rebalance_dates=[pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-28")],
        top_k=1,
        shift_days=1,
        pricing_data=pricing_data,
    )

    assert positions["entry_date"].tolist() == ["20200203", "20200302"]
    assert positions["symbol"].tolist() == ["B", "A"]


def test_execution_sim_positions_align_to_backtest_periods() -> None:
    positions = pd.DataFrame(
        {
            "rebalance_date": [20200131, "2020-02-07", pd.Timestamp("2020-02-14")],
            "entry_date": ["20200203", "20200210", "20200217"],
            "symbol": ["A", "B", "C"],
            "weight": [1 / 3, 1 / 3, 1 / 3],
        }
    )
    period_info = [
        {"rebalance_date": pd.Timestamp("2020-01-31")},
        {"rebalance_date": "20200214"},
    ]

    filtered = _filter_positions_to_backtest_periods(positions, period_info)

    assert filtered is not None
    assert filtered["symbol"].tolist() == ["A", "C"]


def test_backtest_hk_connect_shift_uses_execution_calendar_closed_dates() -> None:
    dates = pd.to_datetime(
        ["2026-04-30", "2026-05-04", "2026-05-05", "2026-05-06", "2026-05-29", "2026-06-01"]
    )
    data = pd.DataFrame(
        {
            "trade_date": dates.repeat(2),
            "symbol": ["A", "B"] * len(dates),
            "pred": [2.0, 1.0] * len(dates),
            "close": [
                100.0,
                100.0,
                100.0,
                100.0,
                100.0,
                100.0,
                110.0,
                90.0,
                120.0,
                80.0,
                130.0,
                70.0,
            ],
        }
    )
    execution = build_execution_model(
        {"calendar": "hk_connect", "closed_dates": ["20260504", "20260505"]},
        default_cost_bps=0,
        default_exit_price_policy="strict",
        default_exit_fallback_policy="ffill",
        default_price_col="close",
    )

    _, _, _, _, period_info = backtest_topk(
        data,
        pred_col="pred",
        price_col="close",
        rebalance_dates=[pd.Timestamp("2026-04-30"), pd.Timestamp("2026-05-29")],
        top_k=1,
        shift_days=1,
        cost_bps=0,
        trading_days_per_year=252,
        execution=execution,
    )

    assert period_info[0]["entry_date"] == pd.Timestamp("2026-05-06")
