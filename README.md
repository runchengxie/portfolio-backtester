# portfolio-backtester

`portfolio-backtester` 是通用组合构造与回测工具包，权威 Python 包是 `portfolio_backtester`。

它接收外部信号、目标持仓和行情数据，提供：

- 组合构造和持仓回放
- 收益、成本和换手分析
- 滑点、交易约束和执行容量模拟
- benchmark、暴露和报告
- 输入输出契约与稳定高层 API

仓库可以独立安装和测试，运行时不依赖 `strategy-pipeline` 或私有研究仓库。

## 安装

```bash
git clone https://github.com/runchengxie/portfolio-backtester.git
cd portfolio-backtester
uv sync --locked --extra dev
uv run --extra dev python -m pytest
```

项目使用 Python 3.12 或更高版本。

## 使用入口

常用高层入口包括：

```python
from portfolio_backtester import (
    PositionBacktestConfig,
    run_position_backtest,
)
```

输入表、最小示例和返回值说明见 [docs/guides/entry-points.md](docs/guides/entry-points.md)。

公开 API 与契约见：

- [公开 API](docs/reference/public-api.md)
- [组合式回测规范](docs/concepts/backtest-spec.md)
- [持仓输出约定](docs/reference/outputs/positions.md)

## 开发检查

```bash
scripts/dev/run_tests.sh lint
scripts/dev/run_tests.sh format
scripts/dev/run_tests.sh typecheck
scripts/dev/run_tests.sh all
scripts/dev/run_tests.sh maintainability
```

`fast` 和 `unit` 是 `all` 的兼容别名，都会运行完整测试集。

BasedPyright 用于补充诊断：

```bash
scripts/dev/run_tests.sh basedpyright
```

详细范围见 [docs/testing.md](docs/testing.md)。

## 仓库边界

本仓库维护通用组合构造、回测、成本、容量、暴露和报告能力。

数据采集、特征工程、模型训练、具体策略规则、任务编排和券商执行由调用方或其他仓库负责。

历史 `cstree.backtesting` 路径由 `strategy-pipeline` 在工作区 1.x 期间提供兼容入口。新代码只使用 `portfolio_backtester`。

## 文档入口

- [文档首页](docs/README.md)
- [常用入口](docs/guides/entry-points.md)
- [成本与执行假设](docs/concepts/execution-costs.md)
- [换手率口径](docs/concepts/turnover.md)
- [成本口径](docs/concepts/cost-breakdown.md)
- [测试和质量检查](docs/testing.md)

请勿提交凭证、账户信息、未授权数据、`artifacts/` 或 `outputs/`。

仓库当前没有许可证文件。公开可见不自动授予复制、修改或再分发权限。
