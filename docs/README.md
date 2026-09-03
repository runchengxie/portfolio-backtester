# 文档入口

本目录记录 `portfolio-backtester` 的输入约定、执行假设、输出契约和开发检查。

## 推荐阅读顺序

1. [根目录 README](../README.md)
2. [常用入口](guides/entry-points.md)
3. [通用多袖组合构造](guides/sleeve-portfolio.md)
4. [组合式回测规范](concepts/backtest-spec.md)
5. [回测后端与统一账本边界](concepts/backend-architecture.md)
6. [机器可读框架状态账本](framework-integration-ledger.yml)
7. [成本与执行假设](concepts/execution-costs.md)
8. [执行容量与每日净值模拟](guides/execution-simulation.md)
9. [风格因子组合权重](concepts/style-factor-portfolio-weighting.md)
10. [AFML 仓位与策略风险](concepts/afml-sizing-and-risk.md)
11. [换手率口径](concepts/turnover.md)
12. [成本口径](concepts/cost-breakdown.md)
13. [回测结果解读](concepts/backtest-interpretation.md)
14. [市场 benchmark 阶梯](concepts/benchmark-ladder.md)
15. [持仓输出约定](reference/outputs/positions.md)
16. [回测输出契约](reference/outputs/backtest-outputs.md)
17. [执行分配参考资产](reference/allocation-reference.md)
18. [公开 API](reference/public-api.md)
19. [测试和质量检查](testing.md)
20. [会计与执行路线图](accounting_execution_roadmap.md)

## 事实来源

| 内容 | 代码位置 |
| --- | --- |
| 顶层公开入口 | `src/portfolio_backtester/__init__.py` |
| 通用多袖组合构造 | `src/portfolio_backtester/sleeve_portfolio.py` |
| 回测规范 | `src/portfolio_backtester/backtest_spec.py` |
| 高层 API | `src/portfolio_backtester/api.py` |
| 输入和输出契约 | `src/portfolio_backtester/contracts.py` |
| 执行领域契约 | `src/portfolio_backtester/execution_contracts.py` |
| 执行分配参考资产 | `src/portfolio_backtester/allocation_reference.py` |
| 后端协议与规范化结果 | `src/portfolio_backtester/backends/` |
| 成本与滑点 | `src/portfolio_backtester/execution.py` |
| 持仓回放 | `src/portfolio_backtester/position_backtest.py` |
| 测试入口 | `scripts/dev/run_tests.sh` |

代码、测试和文档发生冲突时，先核对当前实现，再在同一个改动中修正说明。

## 文档边界

本仓库记录通用组合构造和回测行为。数据下载、因子研究、模型训练、具体策略规则、任务编排和券商下单由调用方负责。

历史迁移记录放在 PR、发布说明或维护记录中。用户指南优先说明当前版本的输入、行为和输出。

## 迁移与历史归属

- [组合回测命名空间](namespace-migration.md)
- [DailyWatch20 组合职责归属](ownership-migration.md)
- [旧仓再资格样本外（OOS）对照桥](guides/incumbent-requalification-oos-controls.md)
