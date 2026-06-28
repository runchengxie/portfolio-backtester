# portfolio-backtester

Portfolio construction and research backtesting package for the research
workspace.

This repository owns `cstree.backtesting.*`: Top-K portfolio construction,
rebalance and execution simulation helpers, capacity/exposure reports,
position post-processing, turnover attribution, benchmark ladders, and
backtest reporting.

Current status: transitional stage-3 split. The package is physically separated
from `cross-sectional-trees`, and workspace gates prevent runtime imports back
into `cstree.pipeline`, `cstree.alpha`, and strategy-pipeline contract helpers.
Full research runs are still orchestrated by `cross-sectional-trees`, but this
package owns the reusable backtesting layer and can consume external signal or
position inputs without importing alpha research internals.

Public package entrypoints include:

- `backtest_topk` for score-to-return Top-K research backtests.
- `StrategySpec` and `construct_positions_from_strategy` for signal-to-position
  construction.
- `PositionBacktestConfig` and `run_position_backtest` for replaying explicit
  target positions against pricing data.

## Local checks

```bash
uv run --extra dev ruff check .
uv run --extra dev ruff format --check .
uv run --extra dev ty check
uv run --extra dev pytest
```

Release/advisory check:

```bash
uv run --extra dev basedpyright
```
