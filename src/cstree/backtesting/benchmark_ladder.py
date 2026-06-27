from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ..metrics import summarize_active_returns

FIELDNAMES = [
    "benchmark_name",
    "role",
    "source_type",
    "expected_market",
    "benchmark_market",
    "strategy_returns_file",
    "benchmark_returns_file",
    "attribution_file",
    "attribution_available",
    "aligned_periods",
    "strategy_total_return",
    "benchmark_total_return",
    "active_total_return",
    "active_mean",
    "tracking_error",
    "information_ratio",
    "beta",
    "alpha",
    "corr",
    "status",
    "error",
]


@dataclass(frozen=True)
class _LadderInputs:
    cfg: dict[str, Any]
    strategy_path: Path
    strategy: pd.Series
    expected_market: str
    periods: pd.DataFrame | None
    periods_per_year: float


@dataclass(frozen=True)
class _BenchmarkEntry:
    role: str
    entry: dict[str, Any]
    benchmark_path: Path | None
    benchmark_market: str


def _resolve_path(path_text: str | Path | None, *, base_dir: Path | None = None) -> Path | None:
    if path_text is None:
        return None
    candidate = Path(path_text).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    if base_dir is not None:
        return (base_dir / candidate).resolve()
    return (Path.cwd() / candidate).resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Benchmark ladder config not found: {path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"Failed to parse benchmark ladder config: {path} ({exc})") from exc
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise SystemExit(f"Benchmark ladder config must be a mapping: {path}")
    return payload


def _section(config: dict[str, Any]) -> dict[str, Any]:
    raw = config.get("benchmark_ladder", config)
    if not isinstance(raw, dict):
        raise SystemExit("benchmark_ladder must be a mapping.")
    return raw


def _return_column(frame: pd.DataFrame, preferred: str | None = None) -> str:
    candidates = [
        preferred,
        "strategy_return",
        "benchmark_return",
        "net_return",
        "return",
        "active_return",
    ]
    for candidate in candidates:
        if candidate and candidate in frame.columns:
            return candidate
    raise ValueError(
        "Returns file must include one return column: strategy_return, "
        "benchmark_return, net_return, return, or active_return."
    )


def _date_column(frame: pd.DataFrame) -> str:
    for candidate in ("trade_date", "date", "period_end", "index"):
        if candidate in frame.columns:
            return candidate
    raise ValueError("Returns file must include a trade_date, date, or period_end column.")


def _parse_dates(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    parsed = pd.to_datetime(text, errors="coerce")
    yyyymmdd = text.str.fullmatch(r"\d{8}")
    if yyyymmdd.any():
        parsed.loc[yyyymmdd] = pd.to_datetime(text.loc[yyyymmdd], format="%Y%m%d", errors="coerce")
    return parsed


def _read_returns(path: Path, *, preferred_return_col: str | None = None) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(f"Returns file not found: {path}")
    frame = pd.read_csv(path)
    date_col = _date_column(frame)
    return_col = _return_column(frame, preferred_return_col)
    series = pd.Series(
        pd.to_numeric(frame[return_col], errors="coerce").to_numpy(dtype=float),
        index=_parse_dates(frame[date_col]),
        name=return_col,
    ).dropna()
    if series.empty:
        raise ValueError(f"Returns file has no usable returns: {path}")
    return series.sort_index()


def _read_periods(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Periods file not found: {path}")
    frame = pd.read_csv(path)
    required = {"entry_date", "exit_date"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Periods file is missing required column(s): {', '.join(missing)}")
    out = frame.copy()
    out["entry_date"] = _parse_dates(out["entry_date"])
    out["exit_date"] = _parse_dates(out["exit_date"])
    out = out.dropna(subset=["entry_date", "exit_date"])
    if out.empty:
        raise ValueError(f"Periods file has no usable entry/exit dates: {path}")
    return out.sort_values("exit_date")


def _compound_daily_returns_to_periods(
    daily_returns: pd.Series,
    periods: pd.DataFrame,
    *,
    include_entry_date: bool = False,
) -> pd.Series:
    daily = daily_returns.copy()
    daily.index = pd.to_datetime(daily.index, errors="coerce")
    daily = daily[daily.index.notna()].sort_index()
    daily = daily.dropna()
    if daily.empty:
        return pd.Series(dtype=float, name=daily_returns.name or "benchmark_return")

    values: list[float] = []
    dates: list[pd.Timestamp] = []
    for period in periods.itertuples(index=False):
        entry_date = pd.Timestamp(period.entry_date)
        exit_date = pd.Timestamp(period.exit_date)
        if include_entry_date:
            window = daily[(daily.index >= entry_date) & (daily.index <= exit_date)]
        else:
            window = daily[(daily.index > entry_date) & (daily.index <= exit_date)]
        if window.empty:
            continue
        values.append(float((1.0 + window.astype(float)).prod() - 1.0))
        dates.append(exit_date)
    return pd.Series(values, index=dates, name=daily_returns.name or "benchmark_return")


def _uses_daily_compounding(entry: dict[str, Any]) -> bool:
    mode = (
        str(entry.get("return_mode") or entry.get("returns_mode") or entry.get("frequency") or "")
        .strip()
        .lower()
    )
    return bool(entry.get("compound_daily_returns")) or mode in {
        "daily",
        "daily_compound",
        "compound_daily",
    }


def _total_return(series: pd.Series) -> float:
    if series.empty:
        return np.nan
    return float((1.0 + series).prod() - 1.0)


def _benchmark_entries(cfg: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    entries: list[tuple[str, dict[str, Any]]] = []
    primary = cfg.get("primary_benchmark") or cfg.get("primary")
    if isinstance(primary, dict):
        entries.append(("primary", primary))
    comparisons = cfg.get("comparisons") or cfg.get("benchmarks") or []
    if not isinstance(comparisons, list):
        raise SystemExit("benchmark_ladder.comparisons must be a list.")
    for item in comparisons:
        if not isinstance(item, dict):
            raise SystemExit("benchmark_ladder.comparisons items must be mappings.")
        entries.append(("comparison", item))
    if not entries:
        raise SystemExit("Benchmark ladder requires primary_benchmark or comparisons.")
    return entries


def _empty_row(
    *,
    role: str,
    entry: dict[str, Any],
    strategy_path: Path,
    benchmark_path: Path | None,
    expected_market: str,
    status: str,
    error: str,
) -> dict[str, Any]:
    attribution_path = _resolve_path(
        entry.get("attribution_file"), base_dir=benchmark_path.parent if benchmark_path else None
    )
    return {
        "benchmark_name": str(entry.get("name") or entry.get("benchmark_name") or ""),
        "role": role,
        "source_type": str(entry.get("source_type") or entry.get("type") or "returns_file"),
        "expected_market": expected_market,
        "benchmark_market": str(entry.get("market") or ""),
        "strategy_returns_file": str(strategy_path),
        "benchmark_returns_file": str(benchmark_path) if benchmark_path else None,
        "attribution_file": str(attribution_path) if attribution_path else None,
        "attribution_available": bool(attribution_path and attribution_path.exists()),
        "status": status,
        "error": error,
    }


def _load_ladder_inputs(config: dict[str, Any], *, config_dir: Path) -> _LadderInputs:
    cfg = _section(config)
    strategy_path = _resolve_path(cfg.get("strategy_returns_file"), base_dir=config_dir)
    if strategy_path is None:
        raise SystemExit("benchmark_ladder.strategy_returns_file is required.")
    periods_per_year = float(cfg.get("periods_per_year") or 12.0)
    strategy_return_col = cfg.get("strategy_return_col")
    strategy = _read_returns(strategy_path, preferred_return_col=strategy_return_col)
    expected_market = str(cfg.get("expected_market") or "").strip()
    periods_path = _resolve_path(
        cfg.get("periods_file") or cfg.get("backtest_periods_file"), base_dir=config_dir
    )
    periods = _read_periods(periods_path) if periods_path is not None else None
    return _LadderInputs(
        cfg=cfg,
        strategy_path=strategy_path,
        strategy=strategy,
        expected_market=expected_market,
        periods=periods,
        periods_per_year=periods_per_year,
    )


def _benchmark_entry(role: str, entry: dict[str, Any], *, config_dir: Path) -> _BenchmarkEntry:
    benchmark_path = _resolve_path(
        entry.get("returns_file") or entry.get("benchmark_returns_file"), base_dir=config_dir
    )
    return _BenchmarkEntry(
        role=role,
        entry=entry,
        benchmark_path=benchmark_path,
        benchmark_market=str(entry.get("market") or "").strip(),
    )


def _market_mismatch_row(candidate: _BenchmarkEntry, inputs: _LadderInputs) -> dict[str, Any]:
    actual = candidate.benchmark_market or "<missing>"
    return _empty_row(
        role=candidate.role,
        entry=candidate.entry,
        strategy_path=inputs.strategy_path,
        benchmark_path=candidate.benchmark_path,
        expected_market=inputs.expected_market,
        status="incompatible",
        error=(
            f"benchmark market {actual} does not match expected market {inputs.expected_market}"
        ),
    )


def _missing_benchmark_row(candidate: _BenchmarkEntry, inputs: _LadderInputs) -> dict[str, Any]:
    return _empty_row(
        role=candidate.role,
        entry=candidate.entry,
        strategy_path=inputs.strategy_path,
        benchmark_path=None,
        expected_market=inputs.expected_market,
        status="unavailable",
        error="missing benchmark returns_file",
    )


def _read_benchmark(candidate: _BenchmarkEntry, inputs: _LadderInputs) -> pd.Series:
    if candidate.benchmark_path is None:
        raise ValueError("missing benchmark returns_file")
    benchmark = _read_returns(
        candidate.benchmark_path,
        preferred_return_col=(
            candidate.entry.get("return_col") or candidate.entry.get("benchmark_return_col")
        ),
    )
    if not _uses_daily_compounding(candidate.entry):
        return benchmark
    if inputs.periods is None:
        raise ValueError(
            "benchmark_ladder.periods_file is required when an entry uses "
            "daily-compounded benchmark returns."
        )
    return _compound_daily_returns_to_periods(
        benchmark,
        inputs.periods,
        include_entry_date=bool(candidate.entry.get("include_entry_date")),
    )


def _no_overlap_row(candidate: _BenchmarkEntry, inputs: _LadderInputs) -> dict[str, Any]:
    return _empty_row(
        role=candidate.role,
        entry=candidate.entry,
        strategy_path=inputs.strategy_path,
        benchmark_path=candidate.benchmark_path,
        expected_market=inputs.expected_market,
        status="incompatible",
        error="no overlapping return dates",
    )


def _ok_row(
    candidate: _BenchmarkEntry,
    inputs: _LadderInputs,
    benchmark: pd.Series,
    active_stats: dict[str, Any],
    *,
    config_dir: Path,
) -> dict[str, Any]:
    aligned = pd.concat(
        [inputs.strategy.rename("strategy"), benchmark.rename("benchmark")],
        axis=1,
    ).dropna()
    attribution_path = _resolve_path(candidate.entry.get("attribution_file"), base_dir=config_dir)
    return {
        **_empty_row(
            role=candidate.role,
            entry=candidate.entry,
            strategy_path=inputs.strategy_path,
            benchmark_path=candidate.benchmark_path,
            expected_market=inputs.expected_market,
            status="ok",
            error="",
        ),
        "attribution_file": str(attribution_path) if attribution_path else None,
        "attribution_available": bool(attribution_path and attribution_path.exists()),
        "aligned_periods": int(active_stats.get("n") or 0),
        "strategy_total_return": _total_return(aligned["strategy"]),
        "benchmark_total_return": _total_return(aligned["benchmark"]),
        "active_total_return": active_stats.get("active_total_return"),
        "active_mean": active_stats.get("mean"),
        "tracking_error": active_stats.get("tracking_error"),
        "information_ratio": active_stats.get("information_ratio"),
        "beta": active_stats.get("beta"),
        "alpha": active_stats.get("alpha"),
        "corr": active_stats.get("corr"),
    }


def _evaluate_benchmark_entry(
    candidate: _BenchmarkEntry,
    inputs: _LadderInputs,
    *,
    config_dir: Path,
) -> dict[str, Any]:
    if inputs.expected_market and candidate.benchmark_market != inputs.expected_market:
        return _market_mismatch_row(candidate, inputs)
    if candidate.benchmark_path is None:
        return _missing_benchmark_row(candidate, inputs)
    try:
        benchmark = _read_benchmark(candidate, inputs)
        active_stats, active = summarize_active_returns(
            inputs.strategy, benchmark, inputs.periods_per_year
        )
        if active.empty:
            return _no_overlap_row(candidate, inputs)
        return _ok_row(candidate, inputs, benchmark, active_stats, config_dir=config_dir)
    except Exception as exc:
        return _empty_row(
            role=candidate.role,
            entry=candidate.entry,
            strategy_path=inputs.strategy_path,
            benchmark_path=candidate.benchmark_path,
            expected_market=inputs.expected_market,
            status="unavailable",
            error=str(exc),
        )


def build_benchmark_ladder(config: dict[str, Any], *, config_dir: Path) -> list[dict[str, Any]]:
    inputs = _load_ladder_inputs(config, config_dir=config_dir)
    rows: list[dict[str, Any]] = []
    for role, entry in _benchmark_entries(inputs.cfg):
        candidate = _benchmark_entry(role, entry, config_dir=config_dir)
        rows.append(_evaluate_benchmark_entry(candidate, inputs, config_dir=config_dir))
    return rows


def write_reports(
    rows: list[dict[str, Any]],
    *,
    output_csv: Path | None,
    output_json: Path | None,
) -> None:
    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(rows, ensure_ascii=True, indent=2, default=str), encoding="utf-8"
        )


def add_benchmark_ladder_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", required=True, help="Benchmark ladder YAML config.")
    parser.add_argument("--output", default=None, help="Output CSV path.")
    parser.add_argument("--output-json", default=None, help="Output JSON path.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging level",
    )
    return parser


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(levelname)s: %(message)s",
    )
    config_path = _resolve_path(args.config)
    assert config_path is not None
    config = _load_yaml(config_path)
    cfg = _section(config)
    rows = build_benchmark_ladder(config, config_dir=config_path.parent)
    output_csv = _resolve_path(
        args.output or cfg.get("output_csv") or cfg.get("output"), base_dir=config_path.parent
    )
    output_json = _resolve_path(
        args.output_json or cfg.get("output_json"), base_dir=config_path.parent
    )
    if output_csv is None and output_json is None:
        print(json.dumps(rows, ensure_ascii=True, indent=2, default=str))
    else:
        write_reports(rows, output_csv=output_csv, output_json=output_json)
        if output_csv:
            logging.info("Benchmark ladder CSV written to %s", output_csv)
        if output_json:
            logging.info("Benchmark ladder JSON written to %s", output_json)
    return rows
