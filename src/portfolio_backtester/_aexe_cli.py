"""A-share executable OOS Top-K: command-line parsing and ``main`` entry point.

``main`` and ``_set_trade_fee_args`` mutate the configuration globals that the
engine functions read. Those globals live on the
:mod:`portfolio_backtester.a_share_executable_oos_topk` shell module, so this
module sets them there at runtime (lazy import avoids a circular import at load
time). Re-exported by that shell.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ._aexe_io import load_positions, load_prices


def _parse_top_ks(text: str) -> list[int]:
    return [int(item) for item in text.split(",") if item]


def _parse_optional_float(value: str) -> float | None:
    if value.lower() in {"", "none", "null", "off"}:
        return None
    return float(value)


def _parse_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "y", "on"}


def _add_trade_fee_args(parser: argparse.ArgumentParser) -> None:
    from .a_share_executable_oos_topk import (
        BUY_COMMISSION_BPS,
        BUY_SLIPPAGE_BPS,
        MIN_COMMISSION_CNY,
        SELL_COMMISSION_BPS,
        SELL_SLIPPAGE_BPS,
        STAMP_TAX_SELL_BPS,
        TRANSFER_FEE_BPS,
    )

    parser.add_argument("--use-detailed-fees", default="false")
    parser.add_argument("--buy-commission-bps", type=float, default=BUY_COMMISSION_BPS)
    parser.add_argument("--sell-commission-bps", type=float, default=SELL_COMMISSION_BPS)
    parser.add_argument("--stamp-tax-sell-bps", type=float, default=STAMP_TAX_SELL_BPS)
    parser.add_argument("--transfer-fee-bps", type=float, default=TRANSFER_FEE_BPS)
    parser.add_argument("--min-commission-cny", type=float, default=MIN_COMMISSION_CNY)
    parser.add_argument("--buy-slippage-bps", type=float, default=BUY_SLIPPAGE_BPS)
    parser.add_argument("--sell-slippage-bps", type=float, default=SELL_SLIPPAGE_BPS)


def _set_trade_fee_args(args: argparse.Namespace) -> None:
    from . import a_share_executable_oos_topk as mod

    mod.USE_DETAILED_FEES = _parse_bool(args.use_detailed_fees)
    mod.BUY_COMMISSION_BPS = float(args.buy_commission_bps)
    mod.SELL_COMMISSION_BPS = float(args.sell_commission_bps)
    mod.STAMP_TAX_SELL_BPS = float(args.stamp_tax_sell_bps)
    mod.TRANSFER_FEE_BPS = float(args.transfer_fee_bps)
    mod.MIN_COMMISSION_CNY = float(args.min_commission_cny)
    mod.BUY_SLIPPAGE_BPS = float(args.buy_slippage_bps)
    mod.SELL_SLIPPAGE_BPS = float(args.sell_slippage_bps)


def main() -> None:
    from . import a_share_executable_oos_topk as mod
    from .a_share_executable_oos_topk import (
        DEFAULT_CAPITAL,
        DEFAULT_COST_BPS,
        DEFAULT_REBALANCE_STRIDE,
        DEFAULT_ROUND_LOT,
        DEFAULT_RUN,
        DEFAULT_TOP_KS,
        IMPACT_BPS_PER_ADV,
        simulate,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN))
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--capital", type=float, default=DEFAULT_CAPITAL)
    parser.add_argument("--round-lot", type=int, default=DEFAULT_ROUND_LOT)
    parser.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS)
    parser.add_argument("--top-ks", default=",".join(str(x) for x in DEFAULT_TOP_KS))
    parser.add_argument("--rebalance-stride", type=int, default=DEFAULT_REBALANCE_STRIDE)
    parser.add_argument("--max-turnover-per-rebalance", default="none")
    parser.add_argument("--hold-buffer-rank", default="none")
    parser.add_argument("--realistic-daily-execution", default="false")
    parser.add_argument("--adv-participation-limit", default="none")
    parser.add_argument("--impact-bps-per-adv", type=float, default=IMPACT_BPS_PER_ADV)
    _add_trade_fee_args(parser)
    args = parser.parse_args()

    mod.CAPITAL = float(args.capital)
    mod.ROUND_LOT = int(args.round_lot)
    mod.COST_BPS = float(args.cost_bps)
    mod.TOP_KS = _parse_top_ks(args.top_ks)
    mod.REBALANCE_STRIDE = max(1, int(args.rebalance_stride))
    mod.MAX_TURNOVER_PER_REBALANCE = _parse_optional_float(args.max_turnover_per_rebalance)
    parsed_buffer = _parse_optional_float(args.hold_buffer_rank)
    mod.HOLD_BUFFER_RANK = int(parsed_buffer) if parsed_buffer is not None else None
    mod.REALISTIC_DAILY_EXECUTION = _parse_bool(args.realistic_daily_execution)
    mod.ADV_PARTICIPATION_LIMIT = _parse_optional_float(args.adv_participation_limit)
    mod.IMPACT_BPS_PER_ADV = float(args.impact_bps_per_adv)
    _set_trade_fee_args(args)

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "live_executable_500k_oos_topk"
    out_dir.mkdir(parents=True, exist_ok=True)
    pos = load_positions(run_dir)
    px = load_prices(sorted(pos["symbol"].unique()), run_dir)
    summaries = []
    all_diag = []
    for top_k in mod.TOP_KS:
        stats, daily, diag, trades = simulate(pos, px, top_k)
        summaries.append(stats)
        all_diag.append(diag)
        daily.to_csv(out_dir / f"daily_top{top_k}.csv", index=False)
        diag.to_csv(out_dir / f"rebalance_diag_top{top_k}.csv", index=False)
        trades.to_csv(out_dir / f"rebalance_trades_top{top_k}.csv", index=False)
        print(
            "done",
            top_k,
            "sharpe",
            round(stats["sharpe"], 3),
            "ret",
            round(stats["total_return"], 3),
            "cash",
            round(stats["avg_cash_weight_daily"], 3),
        )
    summary = pd.DataFrame(summaries).sort_values("top_k")
    summary.to_csv(out_dir / "topk_summary.csv", index=False)
    pd.concat(all_diag, ignore_index=True).to_csv(out_dir / "rebalance_diag_all.csv", index=False)
    meta = {
        "source_run": str(run_dir),
        "positions_source": "positions_by_rebalance_oos.csv",
        "capital": mod.CAPITAL,
        "round_lot": mod.ROUND_LOT,
        "cost_bps": mod.COST_BPS,
        "rebalance_stride": mod.REBALANCE_STRIDE,
        "max_turnover_per_rebalance": mod.MAX_TURNOVER_PER_REBALANCE,
        "hold_buffer_rank": mod.HOLD_BUFFER_RANK,
        "realistic_daily_execution": mod.REALISTIC_DAILY_EXECUTION,
        "adv_participation_limit": mod.ADV_PARTICIPATION_LIMIT,
        "impact_bps_per_adv": mod.IMPACT_BPS_PER_ADV,
        "use_detailed_fees": mod.USE_DETAILED_FEES,
        "buy_commission_bps": mod.BUY_COMMISSION_BPS,
        "sell_commission_bps": mod.SELL_COMMISSION_BPS,
        "stamp_tax_sell_bps": mod.STAMP_TAX_SELL_BPS,
        "transfer_fee_bps": mod.TRANSFER_FEE_BPS,
        "min_commission_cny": mod.MIN_COMMISSION_CNY,
        "buy_slippage_bps": mod.BUY_SLIPPAGE_BPS,
        "sell_slippage_bps": mod.SELL_SLIPPAGE_BPS,
        "affordability_filter": (
            "drop candidate if one 100-share lot exceeds equal target slot at "
            "current equity; backfill from ranks available in source top15"
        ),
        "cash_redistribution": (
            "floor lots, then add one lot at a time to most-underweight "
            "selected names without breaching min(abs 18%, 1/k * 1.35)"
        ),
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("WROTE", out_dir)
