# portfolio-backtester

`portfolio-backtester` 是一个面向研究场景的 Python 组合构造与回测工具包。它接收外部信号、目标持仓和行情数据，完成组合构造、收益回放、交易成本估算、换手分析、容量分析和暴露分析。

仓库可以独立安装和测试。完整研究流程可以在其他项目中调用本包，本包运行时不依赖私有的信号研究仓库或策略编排仓库。

## 主要功能

- 根据模型分数构造 Top-K 多头或多空组合
- 设置持仓缓冲区、分组数量上限、流动性筛选和换手上限
- 从 `StrategySpec` 生成标准目标持仓
- 回放已有目标持仓，支持不同的开仓价和退出价
- 估算固定基点成本、分方向费用和参与率滑点
- 处理停牌或不可交易状态下的退出价格
- 生成收益、换手、成本、容量和暴露分析结果
- 构造专用的风格复制组合，并计算持仓变化和暴露摘要

## 环境要求

- Python 3.12 或更高版本
- `uv`

## 安装

```bash
git clone https://github.com/runchengxie/portfolio-backtester.git
cd portfolio-backtester
uv sync --locked --extra dev
```

安装完成后，可以先运行完整测试：

```bash
uv run --extra dev pytest
```

## 快速开始

下面的示例回放一组已有目标持仓。输入使用内存中的 `DataFrame`，实际项目也可以从 CSV 或 Parquet 文件读取。

```python
import pandas as pd

from cstree.backtesting import PositionBacktestConfig, run_position_backtest

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
| --- | --- |
| `net_returns` | 扣除成本后的分期收益 |
| `gross_returns` | 扣除成本前的分期收益 |
| `periods` | 每个持有期的价格、收益、换手和成本明细 |
| `summary` | 配置快照和汇总统计 |

## 三种常用入口

### 1. 从分数直接运行 Top-K 回测

使用 `backtest_topk`。它支持多头、多空、持仓缓冲、分组约束、流动性筛选、换手上限、独立定价数据和执行模型。

```python
from cstree.backtesting import backtest_topk
```

该函数参数较多，建议先准备以下字段：

- `trade_date`
- `symbol`
- 分数列，例如 `signal`
- 价格列，例如 `close`
- 执行模型需要的流动性列或可交易标记

### 2. 先构造持仓，再单独回测

使用 `StrategySpec` 和 `construct_positions_from_strategy` 生成标准持仓，再把结果交给 `run_position_backtest`。这种方式便于检查目标持仓、保存中间结果和复用定价逻辑。

```python
from cstree.backtesting import StrategySpec, construct_positions_from_strategy
```

### 3. 回放已有目标持仓

使用 `PositionBacktestConfig` 和 `run_position_backtest`。这种方式适合从其他模型、优化器或人工流程接收持仓。

## 输入约定

### 目标持仓

标准文件名为 `positions_by_rebalance.csv`。必需字段如下：

| 字段 | 含义 |
| --- | --- |
| `rebalance_date` | 调仓信号对应的日期 |
| `symbol` | 证券代码 |
| `weight` | 目标权重 |

常用可选字段包括 `entry_date`、`side`、`signal` 和 `rank`。完整说明见 [持仓输出约定](docs/reference/outputs/positions.md)。

### 定价数据

基础字段通常包括：

| 字段 | 含义 |
| --- | --- |
| `trade_date` | 交易日期 |
| `symbol` | 证券代码 |
| 价格列 | 例如 `close`、`open` 或调用方准备的其他价格列 |

执行模型可能还需要成交额、流动性代理和可交易标记。成本、滑点和退出价格的说明见 [成本与执行假设](docs/concepts/execution-costs.md)。

### 持有期

`run_position_backtest` 需要一个持有期表，至少包含：

- `rebalance_date`
- `entry_date`
- `exit_date`

## 公开入口

下面这些对象可以直接从 `cstree.backtesting` 导入：

| 类别 | 入口 |
| --- | --- |
| Top-K 回测 | `backtest_topk` |
| 策略和持仓构造 | `StrategySpec`、`GroupCap`、`strategy_from_config`、`construct_positions_from_strategy` |
| 持仓回放 | `PositionBacktestConfig`、`PositionBacktestResult`、`run_position_backtest` |
| 持仓契约 | `POSITIONS_BY_REBALANCE_CONTRACT`、`validate_positions_by_rebalance_frame`、`assert_positions_by_rebalance_frame` |
| 成本与滑点 | `DetailedTradeFeeModel`、`l2_price_tiered_slippage` |
| 收益汇总 | `summarize_period_returns` |
| 风格复制组合 | `StyleReplicaPortfolioConfig`、`build_style_replica_positions`、`compute_daily_changes`、`compute_daily_exposure`、`compute_style_exposure_summary` |

未列在顶层导出中的模块仍可供仓库内部使用，其接口稳定性低于上表中的公开入口。

## 项目边界

本包从调用方接收信号、持仓和行情数据。数据下载、因子研究、模型训练、任务编排和实盘下单由调用方或其他项目负责。

仓库曾经作为更大研究工作区的一部分维护，目前已经移除对私有数据仓库的运行时依赖。其他项目仍可把它作为子模块或普通 Python 依赖使用。

## 文档

- [文档入口](docs/README.md)
- [成本与执行假设](docs/concepts/execution-costs.md)
- [风格复制组合构造器](docs/concepts/style-replica-portfolio.md)
- [持仓输出约定](docs/reference/outputs/positions.md)
- [测试和质量检查](docs/testing.md)

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

各命令的实际检查范围见 [测试和质量检查](docs/testing.md)。

## 数据与凭证

请勿提交数据提供商凭证、本地环境文件、账户信息或未授权的数据文件。仓库已经忽略 `.env`、`.env.*`、`.envrc`、`artifacts/` 和 `outputs/`，提交前仍应检查暂存区和 Git 历史。

本仓库只提供研究工具，不构成投资建议，也不保证回测结果可以在真实交易中复现。

## 许可证

本仓库当前未附带开源许可证。公开可见不等同于授予复制、修改或再分发权限。计划接受外部使用或贡献时，应先补充明确的许可证。
