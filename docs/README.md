# portfolio-backtester 文档入口

本目录用于承接组合构造和研究回测层文档。当前仓库的详细说明仍以根目录 [README.md](../README.md) 为入口。

## 文档归属

适合放在本仓库的主题：

- Top-K 组合构造、buffer、分组约束、手数约束和持仓后处理。
- 回测收益、交易成本、换手、容量、暴露、benchmark ladder 和报告字段。
- `positions_by_rebalance.csv`、`positions_current*.csv` 及其下游消费约定。
- A 股 round-lot 可执行性、执行模拟和组合层敏感性分析。

仍留在 `strategy-pipeline` 的文档应聚焦编排、CLI、配置合成、运行目录和执行目标导出。后续从 `strategy-pipeline/docs/` 迁移回测主题时，先在原位置保留跳转说明，再更新相对链接和测试。

## 已承接页面

这些页面已经从 `strategy-pipeline` 迁入，并由本仓库维护：

- [concepts/execution-costs.md](concepts/execution-costs.md)
- [reference/outputs/positions.md](reference/outputs/positions.md)

## 后续优先承接内容

后续从 `strategy-pipeline` 拆分文档时，优先迁入：

- `strategy-pipeline/docs/reference/outputs/full-reference.md` 中的 backtest、positions、execution simulation、capacity、exposure 和 benchmark ladder 字段
- `strategy-pipeline/docs/metrics.md` 中的回测收益、成本、换手、容量、暴露和 benchmark ladder 内容
- `strategy-pipeline/docs/concepts/benchmark-protocol.md` 中的 benchmark ladder 和组合层比较内容
- `strategy-pipeline/docs/capabilities.md` 中 `cstree backtest ...` 命令的细节说明
