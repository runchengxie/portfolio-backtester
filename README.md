# portfolio-backtester

`portfolio-backtester` 是面向量化研究的组合构造与回测工具包。它接收外部信号、目标持仓和行情数据，完成组合构造、收益回放、交易成本估算、换手分析、容量分析和暴露分析。

权威 Python 包是 `portfolio_backtester`。仓库可以独立安装和测试，运行时不依赖 `strategy-pipeline` 或私有研究仓库。

## 环境要求

- Python 3.12 或更高版本
- `uv`

## 安装

```bash
git clone https://github.com/runchengxie/portfolio-backtester.git
cd portfolio-backtester
uv sync --locked --extra dev
```

安装后运行测试：

```bash
uv run --extra dev python -m pytest
```

## 快速开始

下面的示例回放一组目标持仓：

```python
import pandas as pd

from portfolio_backtester import PositionBacktestConfig, run_position_backtest

positions = pd.DataFrame(
    [
        {"rebalance_date": "20260102", "symbol": "AAA", "weight": 0.6},
        {"rebalance_date": "20260102", "symbol": "BBB", "weight": 0.4},
    ]
)

pricing = pd.DataFrame(
    [
        {"trade_date": "20260105", "symbol": "AAA", "close": 10.0},
        {"trade_date": "20260105", "symbol": "BBB", "close": 20.0},
        {"trade_date": "20260106", "symbol": "AAA", "close": 10.5},
        {"trade_date": "20260106", "symbol": "BBB", "close": 19.0},
    ]
)

periods = pd.DataFrame(
    [
        {
            "rebalance_date": "20260102",
            "entry_date": "20260105",
            "exit_date": "20260106",
        }
    ]
)

result = run_position_backtest(
    positions=positions,
    pricing=pricing,
    periods=periods,
    config=PositionBacktestConfig(
        price_col="close",
        transaction_cost_bps=10.0,
    ),
)

print(result.periods)
print(result.summary["stats"])
```

`run_position_backtest` 返回 `PositionBacktestResult`：

| 字段 | 内容 |
| --- | --- |
| `net_returns` | 扣除成本后的分期收益 |
| `gross_returns` | 扣除成本前的分期收益 |
| `periods` | 持有期价格、收益、换手和成本明细 |
| `summary` | 配置快照和汇总统计 |

更多调用方式见 [docs/guides/entry-points.md](docs/guides/entry-points.md)。

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

以下内容由调用方或其他仓库负责：

- 数据采集与发布
- 特征工程和模型训练
- 具体策略规则
- 任务编排
- 券商下单和实盘风控

工作区 2.0 已删除旧共享 namespace 和 facade。新代码只使用 `portfolio_backtester`。

## Python namespace

The canonical package is `portfolio_backtester`. Workspace 2.0 has removed the
1.x compatibility namespace and facade; all imports, contracts, artifact types,
logger names, and environment variables are now owner-native.

## 文档入口

- [文档首页](docs/README.md)
- [常用入口](docs/guides/entry-points.md)
- [组合式回测规范](docs/concepts/backtest-spec.md)
- [成本与执行假设](docs/concepts/execution-costs.md)
- [换手率口径](docs/concepts/turnover.md)
- [成本口径](docs/concepts/cost-breakdown.md)
- [持仓输出约定](docs/reference/outputs/positions.md)
- [公开 API](docs/reference/public-api.md)
- [测试和质量检查](docs/testing.md)

## 数据与许可证

请勿提交凭证、账户信息、未授权数据、`artifacts/` 或 `outputs/`。

仓库当前没有许可证文件。公开可见不自动授予复制、修改或再分发权限。
