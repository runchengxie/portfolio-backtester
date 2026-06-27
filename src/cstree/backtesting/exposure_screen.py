from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

STYLE_FACTORS = ("size", "value", "quality", "momentum", "low_vol", "beta")
FIELDNAMES = [
    "status",
    "check",
    "rebalance_date",
    "entry_date",
    "name",
    "metric",
    "value",
    "limit",
]


def _resolve_path(path_text: str | Path | None, *, base_dir: Path | None = None) -> Path | None:
    if path_text is None:
        return None
    candidate = Path(path_text).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    if base_dir is not None:
        return (base_dir / candidate).resolve()
    return (Path.cwd() / candidate).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"Summary JSON must contain an object: {path}")
    return payload


def _nested_mapping(payload: dict[str, Any], *keys: str) -> dict[str, Any]:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return {}
        current = current.get(key)
    return current if isinstance(current, dict) else {}


def _active_summary_from_summary(summary_path: Path) -> Path | None:
    payload = _read_json(summary_path)
    exposure = _nested_mapping(payload, "final_oos", "backtest", "exposure")
    if not exposure:
        exposure = _nested_mapping(payload, "backtest", "exposure")
    return _resolve_path(exposure.get("active_summary_file"), base_dir=summary_path.parent)


def _finite_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _row_id(row: pd.Series) -> dict[str, object]:
    return {
        "rebalance_date": row.get("rebalance_date"),
        "entry_date": row.get("entry_date"),
    }


def _check_row(
    *,
    row: pd.Series,
    check: str,
    name: str,
    metric: str,
    value: float | None,
    limit: float,
    passed: bool,
) -> dict[str, object]:
    return {
        **_row_id(row),
        "status": "passed" if passed else "breached",
        "check": check,
        "name": name,
        "metric": metric,
        "value": value,
        "limit": limit,
    }


def _screen_rows(
    frame: pd.DataFrame,
    *,
    max_abs_style_equal: float,
    max_abs_style_cap: float,
    min_style_coverage: float,
    max_abs_industry_active: float,
    max_industry_weight: float,
) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        for factor in STYLE_FACTORS:
            coverage = _finite_float(row.get(f"{factor}_weight_coverage"))
            if coverage is not None:
                checks.append(
                    _check_row(
                        row=row,
                        check="style_coverage",
                        name=factor,
                        metric="weight_coverage",
                        value=coverage,
                        limit=min_style_coverage,
                        passed=coverage >= min_style_coverage,
                    )
                )
            for metric, limit in (
                ("active_net_vs_equal", max_abs_style_equal),
                ("active_net_vs_cap", max_abs_style_cap),
            ):
                value = _finite_float(row.get(f"{factor}_{metric}"))
                if value is None:
                    continue
                checks.append(
                    _check_row(
                        row=row,
                        check="style_active",
                        name=factor,
                        metric=metric,
                        value=value,
                        limit=limit,
                        passed=abs(value) <= limit,
                    )
                )
        for idx in (1, 2, 3):
            name = str(row.get(f"industry_top_{idx}_name") or "").strip()
            active = _finite_float(row.get(f"industry_top_{idx}_active"))
            if active is not None:
                checks.append(
                    _check_row(
                        row=row,
                        check="industry_active",
                        name=name or f"top_{idx}",
                        metric=f"industry_top_{idx}_active",
                        value=active,
                        limit=max_abs_industry_active,
                        passed=abs(active) <= max_abs_industry_active,
                    )
                )
            weight = _finite_float(row.get(f"industry_top_{idx}_portfolio_net_weight"))
            if weight is not None:
                checks.append(
                    _check_row(
                        row=row,
                        check="industry_weight",
                        name=name or f"top_{idx}",
                        metric=f"industry_top_{idx}_portfolio_net_weight",
                        value=weight,
                        limit=max_industry_weight,
                        passed=abs(weight) <= max_industry_weight,
                    )
                )
    return checks


def build_exposure_screen(
    *,
    summary_file: str | Path | None = None,
    active_summary_file: str | Path | None = None,
    scope: str = "latest",
    max_abs_style_equal: float = 1.0,
    max_abs_style_cap: float = 2.0,
    min_style_coverage: float = 0.8,
    max_abs_industry_active: float = 0.20,
    max_industry_weight: float = 0.30,
) -> dict[str, Any]:
    summary_path = _resolve_path(summary_file)
    active_path = _resolve_path(active_summary_file)
    if active_path is None and summary_path is not None:
        active_path = _active_summary_from_summary(summary_path)
    if active_path is None:
        raise SystemExit("--summary or --active-summary-file is required.")
    if not active_path.exists():
        raise SystemExit(f"Active exposure summary file not found: {active_path}")

    frame = pd.read_csv(active_path)
    if frame.empty:
        raise SystemExit(f"Active exposure summary has no rows: {active_path}")
    if scope == "latest":
        frame = frame.tail(1).copy()
    elif scope != "all":
        raise SystemExit("--scope must be latest or all.")
    checks = _screen_rows(
        frame,
        max_abs_style_equal=max_abs_style_equal,
        max_abs_style_cap=max_abs_style_cap,
        min_style_coverage=min_style_coverage,
        max_abs_industry_active=max_abs_industry_active,
        max_industry_weight=max_industry_weight,
    )
    breached = [check for check in checks if check["status"] == "breached"]
    return {
        "schema_version": 1,
        "status": "passed" if not breached else "breached",
        "summary_file": str(summary_path) if summary_path else None,
        "active_summary_file": str(active_path),
        "scope": scope,
        "rows_checked": int(frame.shape[0]),
        "thresholds": {
            "max_abs_style_equal": max_abs_style_equal,
            "max_abs_style_cap": max_abs_style_cap,
            "min_style_coverage": min_style_coverage,
            "max_abs_industry_active": max_abs_industry_active,
            "max_industry_weight": max_industry_weight,
        },
        "checks": checks,
        "breach_count": len(breached),
    }


def write_csv(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(payload["checks"])


def add_exposure_screen_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--summary", help="Run summary.json containing exposure file paths.")
    parser.add_argument("--active-summary-file", help="Explicit active exposure summary CSV.")
    parser.add_argument("--scope", choices=["latest", "all"], default="latest")
    parser.add_argument("--max-abs-style-equal", type=float, default=1.0)
    parser.add_argument("--max-abs-style-cap", type=float, default=2.0)
    parser.add_argument("--min-style-coverage", type=float, default=0.8)
    parser.add_argument("--max-abs-industry-active", type=float, default=0.20)
    parser.add_argument("--max-industry-weight", type=float, default=0.30)
    parser.add_argument("--out", help="Optional JSON output path. Default: stdout.")
    parser.add_argument("--csv-out", help="Optional CSV checks output path.")
    parser.add_argument("--fail-on-breach", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    payload = build_exposure_screen(
        summary_file=args.summary,
        active_summary_file=args.active_summary_file,
        scope=args.scope,
        max_abs_style_equal=args.max_abs_style_equal,
        max_abs_style_cap=args.max_abs_style_cap,
        min_style_coverage=args.min_style_coverage,
        max_abs_industry_active=args.max_abs_industry_active,
        max_industry_weight=args.max_industry_weight,
    )
    text = json.dumps(payload, ensure_ascii=True, indent=2, default=str)
    out_path = _resolve_path(args.out)
    if out_path is None:
        print(text)
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
    csv_path = _resolve_path(args.csv_out)
    if csv_path is not None:
        write_csv(payload, csv_path)
    if args.fail_on_breach and payload["status"] != "passed":
        raise SystemExit(2)
    return payload
