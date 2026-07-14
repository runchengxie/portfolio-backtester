# 文档入口

本目录说明 `portfolio-backtester` 的输入约定、执行假设、输出契约和开发检查。

第一次接触项目时，建议按下面的顺序阅读：

1. [根目录 README](../README.md)
2. [常用入口](guides/entry-points.md)
3. [组合式回测规范](concepts/backtest-spec.md)
4. [成本与执行假设](concepts/execution-costs.md)
5. [换手率口径](concepts/turnover.md)
6. [成本口径](concepts/cost-breakdown.md)
7. [持仓输出约定](reference/outputs/positions.md)
8. [公开入口](reference/public-api.md)
9. [测试和质量检查](testing.md)

## 文档导航

| 页面 | 主要内容 |
| --- | --- |
| [根目录 README](../README.md) | 项目用途、安装方式、快速示例和文档导航 |
| [常用入口](guides/entry-points.md) | 四种调用方式的详细示例 |
| [组合式回测规范](concepts/backtest-spec.md) | `BacktestSpec`、配置序列化和历史入口迁移 |
| [成本与执行假设](concepts/execution-costs.md) | 成本模型、滑点模型、价格选择和适用边界 |
| [换手率口径](concepts/turnover.md) | `name_turnover` 与 `one_way_turnover` 的定义和公式 |
| [成本口径](concepts/cost-breakdown.md) | `CostBreakdown` 的分项字段说明 |
| [持仓输出约定](reference/outputs/positions.md) | `positions_by_rebalance.csv` 的字段和校验规则 |
| [公开入口](reference/public-api.md) | 完整的顶层公开 API 列表 |
| [测试和质量检查](testing.md) | 本地命令、CI 阻塞项和实际检查范围 |

## 事实来源

文档中的接口、字段和默认值应与以下位置保持一致：

- 顶层公开入口：`src/portfolio_backtester/__init__.py`
- 分数驱动回测规范：`src/portfolio_backtester/backtest_spec.py`
- 分数驱动公开入口：`src/portfolio_backtester/api.py`
- 输入和输出契约：`src/portfolio_backtester/contracts.py`
- 成本与滑点：`src/portfolio_backtester/execution.py`
- 持仓回放：`src/portfolio_backtester/position_backtest.py`
- 测试入口：`scripts/dev/run_tests.sh`
- CI 配置：`.github/workflows/tests.yml`

代码、测试和文档发生冲突时，应先确认当前实现，再在同一个 PR 中修正不一致之处。

## 文档边界

本仓库记录通用组合构造和回测层的行为。数据下载、因子研究、模型训练、具体策略规则、任务编排和实盘下单由调用方负责。

跨仓库的历史迁移记录可以保留在 PR、发布说明或专门的维护记录中。用户指南应优先说明当前版本可以做什么、需要哪些输入、会产生哪些结果。
