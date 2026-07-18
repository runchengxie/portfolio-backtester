# 文档入口

本目录记录 `portfolio-backtester` 的输入约定、执行假设、输出契约和开发检查。

## 推荐阅读顺序

1. [根目录 README](../README.md)
2. [常用入口](guides/entry-points.md)
3. [组合式回测规范](concepts/backtest-spec.md)
4. [回测后端与统一账本边界](concepts/backend-architecture.md)
5. [机器可读框架集成账本](framework-integration-ledger.yml)
6. [成本与执行假设](concepts/execution-costs.md)
7. [执行容量与每日净值模拟](guides/execution-simulation.md)
8. [AFML 仓位与策略风险](concepts/afml-sizing-and-risk.md)
9. [换手率口径](concepts/turnover.md)
10. [成本口径](concepts/cost-breakdown.md)
11. [持仓输出约定](reference/outputs/positions.md)
12. [公开 API](reference/public-api.md)
13. [测试和质量检查](testing.md)
14. [会计与执行路线图](accounting_execution_roadmap.md)

## 事实来源

| 内容 | 代码位置 |
| --- | --- |
| 顶层公开入口 | `src/portfolio_backtester/__init__.py` |
| 回测规范 | `src/portfolio_backtester/backtest_spec.py` |
| 高层 API | `src/portfolio_backtester/api.py` |
| 输入和输出契约 | `src/portfolio_backtester/contracts.py` |
| 执行领域契约 | `src/portfolio_backtester/execution_contracts.py` |
| 后端协议与 canonical result | `src/portfolio_backtester/backends/` |
| 成本与滑点 | `src/portfolio_backtester/execution.py` |
| 持仓回放 | `src/portfolio_backtester/position_backtest.py` |
| 测试入口 | `scripts/dev/run_tests.sh` |

代码、测试和文档发生冲突时，先核对当前实现，再在同一个改动中修正说明。

## 文档边界

本仓库记录通用组合构造和回测行为。数据下载、因子研究、模型训练、具体策略规则、任务编排和券商下单由调用方负责。

历史迁移记录放在 PR、发布说明或维护记录中。用户指南优先说明当前版本的输入、行为和输出。
