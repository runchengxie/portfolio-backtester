# portfolio-backtester

组合构造和研究回测包。

本仓库负责 `cstree.backtesting.*`。它承载 Top-K 组合构造、调仓逻辑、执行模拟、
容量和暴露报告、持仓后处理、换手归因、benchmark 阶梯和回测报告。

当前状态：本仓库已经从原研究仓库中拆出，并作为 `research-workspace` 的子模块锁定版本。
完整研究运行仍由 `strategy-pipeline` 编排；本仓库只负责可复用的组合和回测层，要求能够消费外部信号或持仓输入，同时不导入 alpha 研究内部实现。

公开入口包括：

- `backtest_topk`，用于从模型分数到收益序列的 Top-K 研究回测。
- `StrategySpec` 和 `construct_positions_from_strategy`，用于把信号转成目标持仓。
- `PositionBacktestConfig` 和 `run_position_backtest`，用于基于价格数据回放已有目标持仓。

## 负责的文档

后续新增或迁移文档时，以下主题应优先放在本仓库：

- Top-K、buffer、分组约束、手数约束和持仓后处理。
- 回测收益、交易成本、换手、容量、暴露、benchmark ladder 和报告字段。
- `positions_by_rebalance.csv`、`positions_current*.csv` 及其下游消费约定。
- A 股 round-lot 可执行性、执行模拟和组合层敏感性分析。

`strategy-pipeline` 文档中仍保留的运行编排页可以链接到这里，具体组合和回测方法说明应逐步迁入本仓库。

## 本地检查

```bash
uv run --extra dev ruff check .
uv run --extra dev ruff format --check .
uv run --extra dev ty check
uv run --extra dev pytest
```

发布前或需要诊断类型债时，再运行：

```bash
uv run --extra dev basedpyright
```
