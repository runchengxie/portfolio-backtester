# portfolio-backtester

面向研究场景的 Python 组合构造与回测工具包。接收外部信号、目标持仓和行情数据，完成组合构造、收益回放、交易成本估算、换手分析、容量分析和暴露分析。

仓库可以独立安装和测试。完整研究流程可以在其他项目中调用本包，本包运行时不依赖私有的信号研究仓库或策略编排仓库。

## 环境要求

- Python 3.12 或更高版本
- `uv`

## 安装

```bash
git clone https://github.com/runchengxie/portfolio-backtester.git
cd portfolio-backtester
uv sync --locked --extra dev
```

安装完成后，可以先运行完整测试确认环境正常：

```bash
uv run --extra dev pytest
```

## 快速开始

下面的示例回放一组已有目标持仓。输入使用内存中的 `DataFrame`，实际项目也可以从 CSV 或 Parquet 文件读取。

```python
import pandas as pd

from portfolio_backtester import PositionBacktestConfig, run_position_backtest

positions = pd.DataFrame(
    [
        {'rebalance_date': '20260102', 'symbol': 'AAA', 'weight': 0.6},
        {'rebalance_date': '20260102', 'symbol': 'BBB', 'weight': 0.4},
    ]
)

pricing = pd.DataFrame(
    [
        {'trade_date': '20260105', 'symbol': 'AAA', 'close': 10.0},
        {'trade_date': '20260105', 'symbol': 'BBB', 'close': 20.0},
        {'trade_date': '20260106', 'symbol': 'AAA', 'close': 10.5},
        {'trade_date': '20260106', 'symbol': 'BBB', 'close': 19.0},
    ]
)

periods = pd.DataFrame(
    [
        {
            'rebalance_date': '20260102',
            'entry_date': '20260105',
            'exit_date': '20260106',
        }
    ]
)

result = run_position_backtest(
    positions=positions,
    pricing=pricing,
    periods=periods,
    config=PositionBacktestConfig(
        price_col='close',
        transaction_cost_bps=10.0,
    ),
)

print(result.periods)
print(result.summary['stats'])
```

`run_position_backtest` 返回 `PositionBacktestResult`，其中包含：

| 字段 | 内容 |
|------|------|
| `net_returns` | 扣除成本后的分期收益 |
| `gross_returns` | 扣除成本前的分期收益 |
| `periods` | 每个持有期的价格、收益、换手和成本明细 |
| `summary` | 配置快照和汇总统计 |

## 进一步阅读

项目提供多种调用方式，从高层 `BacktestSpec` 规范到低层持仓回放。以下文档按阅读顺序排列：

| 文档 | 内容 |
|------|------|
| [常用入口](docs/guides/entry-points.md) | 四种调用方式的详细示例 |
| [组合式回测规范](docs/concepts/backtest-spec.md) | `BacktestSpec`、配置序列化和历史入口迁移 |
| [成本与执行假设](docs/concepts/execution-costs.md) | 成本模型、滑点模型、价格选择和适用边界 |
| [换手率口径](docs/concepts/turnover.md) | `name_turnover` 与 `one_way_turnover` 的定义和公式 |
| [成本口径](docs/concepts/cost-breakdown.md) | `CostBreakdown` 的分项字段说明 |
| [持仓输出约定](docs/reference/outputs/positions.md) | `positions_by_rebalance.csv` 的字段和校验规则 |
| [公开入口](docs/reference/public-api.md) | 完整的顶层公开 API 列表 |
| [测试和质量检查](docs/testing.md) | 本地命令、CI 阻塞项和实际检查范围 |
| [文档入口](docs/README.md) | 文档导航和事实来源说明 |

## 开发检查

```bash
scripts/dev/run_tests.sh lint
scripts/dev/run_tests.sh format
scripts/dev/run_tests.sh typecheck
scripts/dev/run_tests.sh all
scripts/dev/run_tests.sh maintainability
```

`BasedPyright` 当前用于补充诊断，并未作为阻塞检查：

```bash
scripts/dev/run_tests.sh basedpyright
```

## 数据与凭证

请勿提交数据提供商凭证、本地环境文件、账户信息或未授权的数据文件。仓库已经忽略 `.env`、`.env.*`、`.envrc`、`artifacts/` 和 `outputs/`，提交前仍应检查暂存区和 Git 历史。

本仓库只提供研究工具，不构成投资建议，也不保证回测结果可以在真实交易中复现。

## 许可证

本仓库当前未附带开源许可证。公开可见不等同于授予复制、修改或再分发权限。计划接受外部使用或贡献时，应先补充明确的许可证。

## Python namespace

The canonical package is `portfolio_backtester`. New code must not add
`cstree.backtesting` imports. The coordinated `strategy-pipeline` compatibility
facade owns the old path during workspace 1.x; removal is scheduled for 2.0.
