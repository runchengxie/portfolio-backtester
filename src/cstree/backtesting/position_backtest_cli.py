from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .position_backtest import (
    PositionBacktestConfig,
    PositionBacktestResult,
    run_position_backtest,
)


def _resolve_path(path_text: str | Path | None, *, base_dir: Path | None = None) -> Path | None:
    if path_text is None or str(path_text).strip() == "":
        return None
    candidate = Path(path_text).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    if base_dir is not None:
        return (base_dir / candidate).resolve()
    return (Path.cwd() / candidate).resolve()


def _read_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Input file not found: {path}")
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _read_yaml(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise SystemExit(f"YAML file must contain a mapping: {path}")
    return payload


def _config_from_run(
    config_path: Path | None,
    *,
    price_col: str | None,
    exit_price_policy: str | None,
    exit_fallback_policy: str | None,
    tradable_col: str | None,
    preserve_gross_exposure: bool | None,
) -> PositionBacktestConfig:
    payload = _read_yaml(config_path)
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    backtest = payload.get("backtest") if isinstance(payload.get("backtest"), dict) else {}
    resolved_exit_policy = str(
        exit_price_policy or backtest.get("exit_price_policy") or "period"
    ).strip()
    if resolved_exit_policy not in {"period", "strict", "ffill", "delay"}:
        raise SystemExit("--exit-price-policy must be one of: period, strict, ffill, delay.")
    resolved_fallback = str(
        exit_fallback_policy or backtest.get("exit_fallback_policy") or "ffill"
    ).strip()
    if resolved_fallback not in {"ffill", "none"}:
        raise SystemExit("--exit-fallback-policy must be one of: ffill, none.")
    preserve = bool(backtest.get("preserve_gross_exposure", False))
    if preserve_gross_exposure is not None:
        preserve = bool(preserve_gross_exposure)
    return PositionBacktestConfig(
        price_col=str(price_col or data.get("price_col") or "close"),
        transaction_cost_bps=float(backtest.get("transaction_cost_bps") or 0.0),
        trading_days_per_year=int(backtest.get("trading_days_per_year") or 252),
        long_only=bool(backtest.get("long_only", True)),
        preserve_gross_exposure=preserve,
        exit_price_policy=resolved_exit_policy,  # type: ignore[arg-type]
        exit_fallback_policy=resolved_fallback,  # type: ignore[arg-type]
        tradable_col=tradable_col or backtest.get("tradable_col"),
    )


def _write_result(
    result: PositionBacktestResult,
    *,
    output_dir: Path,
    output_prefix: str,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "net_file": output_dir / f"{output_prefix}_net.csv",
        "gross_file": output_dir / f"{output_prefix}_gross.csv",
        "periods_file": output_dir / f"{output_prefix}_periods.csv",
        "summary_file": output_dir / f"{output_prefix}_summary.json",
    }
    result.net_returns.to_csv(files["net_file"], index=False)
    result.gross_returns.to_csv(files["gross_file"], index=False)
    result.periods.to_csv(files["periods_file"], index=False)
    files["summary_file"].write_text(
        json.dumps(result.summary, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return files


def _relative_to_run(path: Path, run_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(run_dir.resolve()))
    except ValueError:
        return str(path.resolve())


def _update_run_summary(
    *,
    run_dir: Path,
    files: dict[str, Path],
    result: PositionBacktestResult,
    replace_run_backtest: bool,
) -> None:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise SystemExit(f"Run summary not found: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    section = {
        "schema": result.summary["schema"],
        "stats": result.summary["stats"],
        "net_file": _relative_to_run(files["net_file"], run_dir),
        "gross_file": _relative_to_run(files["gross_file"], run_dir),
        "periods_file": _relative_to_run(files["periods_file"], run_dir),
        "summary_file": _relative_to_run(files["summary_file"], run_dir),
    }
    summary["position_backtest"] = section
    if replace_run_backtest:
        backtest = summary.setdefault("backtest", {})
        backtest["stats"] = result.summary["stats"]
        backtest["net_file"] = section["net_file"]
        backtest["gross_file"] = section["gross_file"]
        backtest["periods_file"] = section["periods_file"]
        backtest["return_source"] = "position_backtest"
        backtest["exit_price_policy"] = result.summary["config"]["exit_price_policy"]
        backtest["exit_fallback_policy"] = result.summary["config"]["exit_fallback_policy"]
        backtest["tradable_col"] = result.summary["config"].get("tradable_col")
        backtest.pop("stats_inherited_from_run", None)
        backtest["inheritance_note"] = (
            "Backtest stats and return files were replaced by explicit "
            "positions_by_rebalance returns via cstree position-backtest."
        )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def add_position_backtest_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--run-dir", required=True, help="Existing run directory.")
    parser.add_argument("--positions-file", help="Positions CSV. Defaults to run positions file.")
    parser.add_argument("--pricing-file", required=True, help="Pricing panel CSV/parquet.")
    parser.add_argument(
        "--periods-file",
        help="Backtest periods CSV. Defaults to run backtest periods.",
    )
    parser.add_argument("--config", help="Config YAML. Defaults to <run-dir>/config.used.yml.")
    parser.add_argument("--price-col", help="Override price column.")
    parser.add_argument(
        "--exit-price-policy",
        choices=["period", "strict", "ffill", "delay"],
        help="Exit price policy. Defaults to config backtest.exit_price_policy or period.",
    )
    parser.add_argument(
        "--exit-fallback-policy",
        choices=["ffill", "none"],
        help="Fallback policy for delay exits. Defaults to config or ffill.",
    )
    parser.add_argument(
        "--tradable-col",
        help="Optional tradable flag column for strict/ffill/delay exits.",
    )
    parser.add_argument(
        "--preserve-gross-exposure",
        action="store_true",
        default=None,
        help=(
            "Preserve sub-100% gross exposure in positions as cash instead of "
            "renormalizing each rebalance to full investment."
        ),
    )
    parser.add_argument("--output-dir", help="Output directory. Defaults to run dir.")
    parser.add_argument("--output-prefix", default="position_backtest")
    parser.add_argument("--update-summary", action="store_true")
    parser.add_argument("--replace-run-backtest", action="store_true")
    return parser


def run(args: argparse.Namespace) -> PositionBacktestResult:
    run_dir = _resolve_path(args.run_dir)
    if run_dir is None:
        raise SystemExit("--run-dir is required.")
    config_path = _resolve_path(args.config, base_dir=run_dir) or run_dir / "config.used.yml"
    positions_path = _resolve_path(args.positions_file, base_dir=run_dir) or run_dir / (
        "positions_by_rebalance.csv"
    )
    periods_path = _resolve_path(args.periods_file, base_dir=run_dir) or run_dir / (
        "backtest_periods.csv"
    )
    pricing_path = _resolve_path(args.pricing_file, base_dir=Path.cwd())
    output_dir = _resolve_path(args.output_dir, base_dir=run_dir) or run_dir
    if pricing_path is None:
        raise SystemExit("--pricing-file is required.")

    config = _config_from_run(
        config_path,
        price_col=args.price_col,
        exit_price_policy=args.exit_price_policy,
        exit_fallback_policy=args.exit_fallback_policy,
        tradable_col=args.tradable_col,
        preserve_gross_exposure=args.preserve_gross_exposure,
    )
    result = run_position_backtest(
        positions=_read_frame(positions_path),
        pricing=_read_frame(pricing_path),
        periods=_read_frame(periods_path),
        config=config,
    )
    files = _write_result(result, output_dir=output_dir, output_prefix=str(args.output_prefix))
    if args.update_summary or args.replace_run_backtest:
        _update_run_summary(
            run_dir=run_dir,
            files=files,
            result=result,
            replace_run_backtest=bool(args.replace_run_backtest),
        )
    print(json.dumps(result.summary, ensure_ascii=False, indent=2, default=str))
    return result
