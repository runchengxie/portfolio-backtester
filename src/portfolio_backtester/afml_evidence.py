"""Generate AFML evidence sidecars from persisted portfolio run artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, SupportsFloat, cast

import numpy as np
import pandas as pd
import yaml

from .evidence_receipts import (
    build_portfolio_sizing_receipt,
    sha256_file,
    write_receipt,
)
from .hrp import HrpConfig, hierarchical_risk_parity
from .strategy_risk import implementation_shortfall_metrics, summarize_strategy_risk


def generate_run_afml_evidence(
    run_dir: str | Path,
    *,
    configuration: Mapping[str, Any] | None = None,
    target_sharpe: float = 1.0,
    evaluation_years: float = 2.0,
    bootstrap_samples: int = 2000,
    random_state: int = 0,
    hrp_returns_path: str | Path | None = None,
) -> dict[str, object]:
    """Write sizing, strategy-risk, and optional HRP evidence sidecars."""

    root = Path(run_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Run directory not found: {root}")
    config = dict(configuration or _load_run_config(root))
    risk_path = _write_strategy_risk_report(
        root,
        target_sharpe=target_sharpe,
        evaluation_years=evaluation_years,
        bootstrap_samples=bootstrap_samples,
        random_state=random_state,
    )
    sizing_path = _write_sizing_receipt(root, config)
    evidence: dict[str, object] = {
        "sizing_receipt": _evidence_entry(sizing_path),
        "strategy_risk": _evidence_entry(risk_path),
    }
    outputs: dict[str, object] = {
        "sizing_receipt": str(sizing_path),
        "strategy_risk_report": str(risk_path),
    }
    if hrp_returns_path is not None:
        hrp_path, weights_path = _write_hrp_receipt(
            root,
            config=config,
            returns_path=hrp_returns_path,
        )
        evidence["hrp_receipt"] = _evidence_entry(hrp_path)
        outputs.update({"hrp_receipt": str(hrp_path), "hrp_weights": str(weights_path)})
    fragment = {
        "schema_version": 1,
        "run_dir": str(root),
        "evidence": evidence,
        "outputs": outputs,
    }
    fragment_path = root / "afml_evidence_fragment.json"
    write_receipt(fragment, fragment_path)
    fragment["fragment_path"] = str(fragment_path)
    return fragment


def maybe_generate_run_afml_evidence(
    *,
    context: Mapping[str, Any],
    artifacts: dict[str, Any],
) -> None:
    """Generate configured AFML evidence and add its paths to output artifacts."""
    config_raw = context.get("config")
    config = config_raw if isinstance(config_raw, Mapping) else {}
    protocol_raw = config.get("research_protocol")
    protocol = protocol_raw if isinstance(protocol_raw, Mapping) else {}
    if not bool(protocol.get("generate_afml_evidence", False)):
        return

    fragment = generate_run_afml_evidence(
        context["run_dir"],
        configuration=config,
        target_sharpe=float(protocol.get("target_sharpe", 1.0)),
        evaluation_years=float(protocol.get("evaluation_years", 2.0)),
        bootstrap_samples=int(protocol.get("bootstrap_samples", 2000)),
        random_state=int(protocol.get("random_state", 0)),
        hrp_returns_path=protocol.get("hrp_returns"),
    )
    outputs = fragment.get("outputs")
    if not isinstance(outputs, Mapping):
        return
    artifacts.update(
        {
            "sizing_receipt_path": outputs.get("sizing_receipt"),
            "strategy_risk_report_path": outputs.get("strategy_risk_report"),
            "hrp_receipt_path": outputs.get("hrp_receipt"),
            "hrp_weights_path": outputs.get("hrp_weights"),
            "afml_evidence_fragment_path": fragment.get("fragment_path"),
        }
    )


def merge_evidence_fragment(
    manifest: Mapping[str, object],
    fragment: Mapping[str, object],
) -> dict[str, object]:
    """Merge generated evidence entries into a protocol manifest."""

    merged = dict(manifest)
    current = merged.get("evidence")
    evidence = dict(current) if isinstance(current, Mapping) else {}
    generated = fragment.get("evidence")
    if isinstance(generated, Mapping):
        evidence.update(generated)
    merged["evidence"] = evidence
    return merged


def _write_strategy_risk_report(
    root: Path,
    *,
    target_sharpe: float,
    evaluation_years: float,
    bootstrap_samples: int,
    random_state: int,
) -> Path:
    net = _read_return_series(root / "backtest_net.csv", "net_return")
    gross = _read_return_series(root / "backtest_gross.csv", "gross_return")
    turnover = _read_return_series(root / "backtest_turnover.csv", "turnover")
    report: dict[str, Any] = dict(
        summarize_strategy_risk(
            net,
            periods_per_year=_infer_periods_per_year(net),
            target_sharpe=target_sharpe,
            evaluation_years=evaluation_years,
            bootstrap_samples=bootstrap_samples,
            random_state=random_state,
        ).to_dict()
    )
    report.update(
        {
            "schema_version": 1,
            "target_sharpe": float(target_sharpe),
            "evaluation_years": float(evaluation_years),
            "bootstrap_samples": int(bootstrap_samples),
            "implementation_shortfall": implementation_shortfall_metrics(
                gross_returns=gross,
                net_returns=net,
                turnover=turnover,
            ),
            "inputs": {
                "backtest_net": "backtest_net.csv",
                "backtest_gross": "backtest_gross.csv",
                "backtest_turnover": "backtest_turnover.csv",
            },
        }
    )
    path = root / "strategy_risk_report.json"
    write_receipt(report, path)
    return path


def _write_sizing_receipt(root: Path, config: Mapping[str, Any]) -> Path:
    positions_path = _resolve_positions_path(root)
    positions = pd.read_csv(positions_path)
    symbol_col = "symbol" if "symbol" in positions.columns else "ticker"
    if symbol_col not in positions.columns or "weight" not in positions.columns:
        raise ValueError(
            f"Positions must contain symbol/ticker and weight columns: {positions_path}"
        )
    weights = pd.to_numeric(positions["weight"], errors="coerce")
    weights.index = positions[symbol_col].astype(str)
    strategy_cfg = _mapping(config.get("strategy"))
    backtest_cfg = _mapping(config.get("backtest"))
    method = str(strategy_cfg.get("weighting", backtest_cfg.get("weighting", "equal")))
    receipt = build_portfolio_sizing_receipt(
        weights,
        method=method,
        configuration={
            "strategy": dict(strategy_cfg),
            "backtest": {
                key: backtest_cfg[key]
                for key in (
                    "top_k",
                    "short_k",
                    "long_only",
                    "buffer_entry",
                    "buffer_exit",
                    "weighting",
                    "max_turnover_per_rebalance",
                )
                if key in backtest_cfg
            },
        },
        source_positions=positions_path,
        calibration_artifact=_optional_artifact(root, "probability_calibration.json"),
        covariance_artifact=_optional_artifact(root, "covariance_snapshot.parquet"),
    )
    path = root / "sizing_receipt.json"
    write_receipt(receipt, path)
    return path


def _write_hrp_receipt(
    root: Path,
    *,
    config: Mapping[str, Any],
    returns_path: str | Path,
) -> tuple[Path, Path]:
    settings = _mapping(config.get("hrp"))
    result = hierarchical_risk_parity(
        _read_returns_matrix(returns_path),
        config=HrpConfig(
            linkage_method=str(settings.get("linkage_method", "single")),
            shrinkage=float(settings.get("shrinkage", 0.0)),
            min_weight=float(settings.get("min_weight", 0.0)),
            max_weight=_optional_float(settings.get("max_weight")),
        ),
    )
    weights_path = root / "hrp_weights.csv"
    result.weights.rename("weight").to_csv(weights_path, index_label="asset")
    receipt = result.receipt()
    receipt.update(
        {
            "returns_file": str(Path(returns_path).expanduser()),
            "returns_sha256": sha256_file(returns_path),
            "weights_file": str(weights_path),
            "weights_sha256": sha256_file(weights_path),
        }
    )
    path = root / "hrp_receipt.json"
    write_receipt(receipt, path)
    return path, weights_path


def _read_return_series(path: Path, value_col: str) -> pd.Series:
    if not path.is_file():
        raise FileNotFoundError(f"Required backtest artifact not found: {path}")
    frame = pd.read_csv(path)
    if "period_end" not in frame.columns or value_col not in frame.columns:
        raise ValueError(f"Invalid backtest return artifact: {path}")
    dates = pd.to_datetime(frame["period_end"], errors="coerce")
    values = pd.to_numeric(frame[value_col], errors="coerce")
    return (
        pd.Series(values.to_numpy(dtype=float), index=dates, name=value_col).dropna().sort_index()
    )


def _read_returns_matrix(path: str | Path) -> pd.DataFrame:
    source = Path(path).expanduser()
    frame = pd.read_csv(source)
    if frame.shape[1] < 3:
        raise ValueError("HRP returns file must contain a date column and at least two assets")
    date_col = "period_end" if "period_end" in frame.columns else str(frame.columns[0])
    frame.index = pd.to_datetime(frame.pop(date_col), errors="coerce")
    frame = frame.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    if frame.shape[1] < 2:
        raise ValueError("HRP returns file must contain at least two numeric return series")
    return frame


def _resolve_positions_path(root: Path) -> Path:
    for name in ("positions_current_live.csv", "positions_current.csv"):
        candidate = root / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No current positions artifact found under: {root}")


def _infer_periods_per_year(values: pd.Series) -> float:
    if isinstance(values.index, pd.DatetimeIndex) and len(values.index) > 1:
        deltas = values.index.to_series().sort_values().diff().dt.days.dropna()
        positive = deltas.loc[deltas > 0]
        if not positive.empty and float(positive.median()) > 0:
            return 365.25 / float(positive.median())
    return 252.0


def _load_run_config(root: Path) -> Mapping[str, Any]:
    path = root / "config.used.yml"
    if not path.is_file():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, Mapping) else {}


def _evidence_entry(path: Path) -> dict[str, object]:
    return {"status": "pass", "path": str(path), "sha256": sha256_file(path)}


def _optional_artifact(root: Path, name: str) -> str | None:
    path = root / name
    return str(path) if path.is_file() else None


def _mapping(value: object) -> Mapping[str, Any]:
    return cast(Mapping[str, Any], value) if isinstance(value, Mapping) else {}


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    number = float(cast(SupportsFloat, value))
    return number if np.isfinite(number) else None


__all__ = [
    "generate_run_afml_evidence",
    "maybe_generate_run_afml_evidence",
    "merge_evidence_fragment",
]
