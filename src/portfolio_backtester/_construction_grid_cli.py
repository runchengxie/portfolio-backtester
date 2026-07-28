from __future__ import annotations

import argparse
import json
import logging
from typing import Any

from . import construction_grid_reports as _construction_grid_reports
from ._construction_grid_eval import _resolve_backtest_topk_fn, build_construction_grid
from ._construction_grid_io import _load_yaml, _resolve_path


def add_construction_grid_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", required=True, help="Construction grid YAML config.")
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
    rows = build_construction_grid(
        config,
        config_dir=config_path.parent,
        backtest_topk_fn=_resolve_backtest_topk_fn(getattr(args, "backtest_topk_fn", None)),
        dynamic_ensemble_fn=getattr(args, "dynamic_ensemble_fn", None),
    )
    cfg = config.get("construction_grid", config)
    output_csv = _resolve_path(
        args.output or cfg.get("output_csv") or cfg.get("output"), base_dir=config_path.parent
    )
    output_json = _resolve_path(
        args.output_json or cfg.get("output_json"), base_dir=config_path.parent
    )
    selection_cfg = cfg.get("rolling_selection") or cfg.get("inertia_selection")
    selection_report = None
    selection_output = None
    if selection_cfg:
        if not isinstance(selection_cfg, dict):
            raise SystemExit("construction_grid.rolling_selection must be a mapping.")
        selection_report = _construction_grid_reports.build_inertia_selection_report(
            rows, selection_cfg
        )
        selection_output = _resolve_path(
            selection_cfg.get("output_json") or selection_cfg.get("output"),
            base_dir=config_path.parent,
        )
    if output_csv is None and output_json is None:
        print(json.dumps(rows, ensure_ascii=True, indent=2, default=str))
    else:
        _construction_grid_reports.write_reports(
            rows, output_csv=output_csv, output_json=output_json
        )
        if output_csv:
            logging.info("Construction grid CSV written to %s", output_csv)
        if output_json:
            logging.info("Construction grid JSON written to %s", output_json)
    if selection_report is not None:
        if selection_output is None:
            print(json.dumps(selection_report, ensure_ascii=True, indent=2, default=str))
        else:
            selection_output.parent.mkdir(parents=True, exist_ok=True)
            selection_output.write_text(
                json.dumps(selection_report, ensure_ascii=True, indent=2, default=str),
                encoding="utf-8",
            )
            logging.info("Construction grid rolling selection JSON written to %s", selection_output)
    return rows
