# portfolio-backtester

Portfolio construction and research backtesting package for the research
workspace.

This repository owns `cstree.backtesting.*`: Top-K portfolio construction,
rebalance and execution simulation helpers, capacity/exposure reports,
position post-processing, turnover attribution, benchmark ladders, and
backtest reporting.

Current status: transitional stage-3 split. The package is physically separated
from `cross-sectional-trees`, but some modules still import shared workspace
helpers from `cstree.pipeline`, `cstree.contracts`, `cstree.alpha`, and other
strategy orchestration modules. Run it from `research-workspace` with the
sibling submodules checked out until those shared interfaces are extracted.

## Local checks

```bash
uv run --extra dev ruff check src
uv run --extra dev ruff format --check src
uv run --extra dev basedpyright
```

