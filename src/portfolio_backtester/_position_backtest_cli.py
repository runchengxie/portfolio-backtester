"""Position backtest command-line entry points (re-exported by ``position_backtest``)."""

from __future__ import annotations

from typing import Any

from ._position_backtest_config import (
    PositionBacktestConfig,
)
from ._position_backtest_engine import run_position_backtest


def add_position_backtest_args(parser: Any) -> Any:
    parser.add_argument(
        "--positions",
        required=True,
        help="CSV with rebalance_date, entry_date, symbol, weight columns",
    )
    parser.add_argument(
        "--pricing",
        required=True,
        help="CSV with trade_date, symbol, price columns",
    )
    parser.add_argument(
        "--periods",
        required=True,
        help="CSV with rebalance_date, entry_date, exit_date columns",
    )
    parser.add_argument("--price-col", default="close", help="Price column name")
    parser.add_argument("--entry-price-col", default=None, help="Entry price column name")
    parser.add_argument("--exit-price-col", default=None, help="Exit price column name")
    parser.add_argument(
        "--transaction-cost-bps",
        type=float,
        default=0.0,
        help="Round-trip transaction cost in basis points",
    )
    parser.add_argument(
        "--trading-days-per-year",
        type=int,
        default=252,
        help="Trading days per year for annualization",
    )
    parser.add_argument(
        "--exit-price-policy",
        default="period",
        choices=["period", "strict", "ffill", "delay"],
        help="Exit price resolution policy",
    )
    parser.add_argument(
        "--exit-fallback-policy",
        default="ffill",
        choices=["ffill", "none"],
        help="Fallback when an exit price is unavailable",
    )
    parser.add_argument("--tradable-col", default=None, help="Tradable flag column name")
    parser.add_argument(
        "--intraday-bars",
        default=None,
        help="Optional intraday bars CSV used to compute VWAP price overrides",
    )
    return parser


def run(args: Any) -> Any:
    import pandas as pd

    positions = pd.read_csv(args.positions)
    pricing = pd.read_csv(args.pricing)
    periods = pd.read_csv(args.periods)
    config = PositionBacktestConfig(
        price_col=args.price_col,
        entry_price_col=args.entry_price_col,
        exit_price_col=args.exit_price_col,
        transaction_cost_bps=args.transaction_cost_bps,
        trading_days_per_year=args.trading_days_per_year,
        exit_price_policy=args.exit_price_policy,
        exit_fallback_policy=args.exit_fallback_policy,
        tradable_col=args.tradable_col,
    )
    intraday_bars = pd.read_csv(args.intraday_bars) if args.intraday_bars else None
    result = run_position_backtest(
        positions=positions,
        pricing=pricing,
        periods=periods,
        config=config,
        intraday_bars=intraday_bars,
    )
    result.periods.to_csv(args.output, index=False) if getattr(args, "output", None) else None
    return result
