# portfolio-backtester

`portfolio-backtester` 是一个面向研究场景的 Python 组合构造与回测工具包。它接收外部信号、目标持仓和行情数据，完成组合构造、收益回放、交易成本估算、换手分析、容量分析和暴露分析。

仓库可以独立安装和测试。完整研究流程可以在其他项目中调用本包，本包运行时不依赖私有的信号研究仓库或策略编排仓库。

## 主要功能

- 根据模型分数构造 Top-K 多头或多空组合
- 设置持仓缓冲区、分组数量上限、流动性筛选和换手上限
- 使用可序列化的 `BacktestSpec` 组合策略、执行假设和回测区间
- 从 `StrategySpec` 生成标准目标持仓
- 回放已有目标持仓，支持不同的开仓价和退出价
- 估算固定基点成本、分方向费用和参与率滑点
- 处理停牌或不可交易状态下的退出价格
- 生成收益、换手、成本、容量和暴露分析结果
- 通过统一后端结果比较原生回放和 Qlib，要求分类并解释语义差异
- 通过版本化 JSON 与外部 LEAN 黄金基准交换场景、结果和成交证据
- 在独立的 `cstree.backtesting.products` 命名空间提供 DailyWatch20 产品选择逻辑

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

基础安装不包含 scikit-learn 或 XGBoost。组合与回测层接收调用方已经计算完成的分数，不负责训练模型。IC 显著性统计直接依赖 SciPy，Parquet 和 YAML 支持继续由 PyArrow 与 PyYAML 提供。

运行 Qlib 差分回测时安装可选依赖：

```bash
uv sync --locked --extra dev --extra qlib
```

Qlib 不属于基础安装。可选依赖把 pandas 限制在 Qlib 已验证的 2.x 范围，并使用 MLflow 2.x 或 3.x，避免 pyqlib 的宽松依赖解析到不兼容组合。LEAN 参考运行通过 JSON 文件在外部进程完成，本包没有 LEAN 运行时依赖。

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

## 常用入口

### 1. 使用组合规范运行回测

新代码建议使用 `BacktestSpec` 和 `run_backtest`。`StrategySpec` 负责选股和权重设置，`ExecutionModel` 负责开仓价、退出规则、成本、滑点和筛选约束。回测区间和调仓设置保存在 `BacktestSpec` 中。

```python
import pandas as pd

from cstree.backtesting import BacktestSpec, StrategySpec, run_backtest
from cstree.backtesting.execution import build_execution_model

# scores 的 DataFrame 构造省略。它至少包含 trade_date、symbol、signal 和 close。
execution = build_execution_model(
    None,
    default_cost_bps=10.0,
    default_exit_price_policy='strict',
    default_exit_fallback_policy='ffill',
    default_price_col='close',
)
spec = BacktestSpec(
    strategy=StrategySpec(
        name='topk-demo',
        type='topk_buffered_long_only',
        score_col='signal',
        top_k=20,
        buffer_exit=5,
        weighting='equal',
    ),
    execution=execution,
    rebalance_dates=(
        pd.Timestamp('2026-01-05'),
        pd.Timestamp('2026-01-12'),
    ),
    shift_days=1,
    trading_days_per_year=252,
)

result = run_backtest(scores, spec)
```

`BacktestSpec.to_mapping()` 可以生成适合写入 JSON 或 YAML 的配置，`BacktestSpec.from_mapping()` 可以恢复规范。行情表不进入配置。信号和定价数据在运行时传给 `run_backtest`。

### 2. 使用历史 Top-K 兼容入口

`backtest_topk` 保留原有签名和默认行为，并把参数转换为 `StrategySpec`、`ExecutionModel` 和 `BacktestSpec` 后调用同一条执行路径。现阶段该入口不会发出弃用警告。

```python
from cstree.backtesting import backtest_topk
```

这两个分数驱动入口通常需要以下字段：

- `trade_date`
- `symbol`
- 分数列，例如 `signal`
- 价格列，例如 `close`
- 执行模型需要的流动性列或可交易标记

### 3. 先构造持仓，再单独回测

使用 `StrategySpec` 和 `construct_positions_from_strategy` 生成标准持仓，再把结果交给 `run_position_backtest`。这种方式便于检查目标持仓、保存中间结果和复用定价逻辑。

```python
from cstree.backtesting import StrategySpec, construct_positions_from_strategy
```

### 4. 回放已有目标持仓

使用 `PositionBacktestConfig` 和 `run_position_backtest`。这种方式适合从其他模型、优化器或人工流程接收持仓。

### 5. 使用产品选择模块

DailyWatch20 是消费预计算分数的产品规则，不属于通用回测内核。新代码从产品命名空间导入：

```python
from cstree.backtesting.products import DailyWatch20Config, select_daily_watch20
```

历史顶层导入继续兼容。`cstree.backtesting.daily_watch20` 模块路径属于弃用兼容层，导入时会发出 `DeprecationWarning`。

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
| 分数驱动回测 | `BacktestSpec`、`run_backtest`、`backtest_topk` |
| 后端和差分 | `BacktestBackend`、`NativeAShareReplayBackend`、`compare_backtest_results` |
| 策略和持仓构造 | `StrategySpec`、`GroupCap`、`strategy_from_config`、`construct_positions_from_strategy` |
| 持仓回放 | `PositionBacktestConfig`、`PositionBacktestResult`、`run_position_backtest` |
| 持仓契约 | `POSITIONS_BY_REBALANCE_CONTRACT`、`validate_positions_by_rebalance_frame`、`assert_positions_by_rebalance_frame` |
| 成本与滑点 | `DetailedTradeFeeModel`、`l2_price_tiered_slippage` |
| 换手与成本 | `TurnoverBreakdown`、`CostBreakdown`、`name_turnover`、`annualize_turnover`、`turnover_from_trade_weights` |
| 收益汇总 | `summarize_period_returns` |
| 产品选择 | `cstree.backtesting.products` 中的 DailyWatch20 API |

未列在顶层导出中的模块仍可供仓库内部使用，其接口稳定性低于上表中的公开入口。

## 换手率口径

项目区分两类容易被混用的换手率：

- `name_turnover`：持仓名称替换比例，适合描述 Top-K 名单稳定性。
- `TurnoverBreakdown.one_way_turnover`：基于目标权重变化的单边换手率，用于成本核算。

`TurnoverBreakdown` 同时保留以下字段，避免只报告一个含义模糊的 `turnover`：

- `buy_weight`
- `sell_weight`
- `gross_traded_weight`
- `half_l1_turnover`
- `one_way_turnover`

对于非初始调仓：

```text
half_l1_turnover = 0.5 * sum(abs(target_weight - drifted_weight))
```

初始建仓沿用历史成本口径，`one_way_turnover` 等于实际买入的 gross exposure；
`half_l1_turnover` 仍保留其严格的数学定义。`annualize_turnover` 只做线性年化，
用于描述交易强度，不代表可复利收益。

## 成本口径

`CostBreakdown` 为回测结果提供统一的费用视图：

- `fee_cost`：显式费用；
- `slippage_cost`：隐式滑点；
- `total_cost`：两者之和。

后续新增佣金、税费、价差、市场冲击或机会成本时，应继续保持分项字段，避免把全部成本压缩成一个无法审计的 bps 数字。

## 项目边界

本包从调用方接收信号、持仓和行情数据。数据下载、因子研究、模型训练、任务编排和实盘下单由调用方或其他项目负责。现有 DailyWatch20 规则位于明确的产品命名空间，不会在导入通用回测包时加载。

仓库曾经作为更大研究工作区的一部分维护，目前已经移除对私有数据仓库的运行时依赖。其他项目仍可把它作为子模块或普通 Python 依赖使用。

## 文档

- [文档入口](docs/README.md)
- [组合式回测规范](docs/concepts/backtest-spec.md)
- [回测后端和差分证据](docs/concepts/backtest-backends.md)
- [DailyWatch20 产品模块](docs/products/daily-watch20.md)
- [成本与执行假设](docs/concepts/execution-costs.md)
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
