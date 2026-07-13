# 常用入口

`portfolio-backtester` 提供四种调用方式，从高层规范到低层持仓回放逐步深入。刚接触项目时，建议从入口 1 开始。

## 1. 使用组合规范运行回测（推荐）

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

输入数据通常需要以下字段：

- `trade_date`
- `symbol`
- 分数列，例如 `signal`
- 价格列，例如 `close`
- 执行模型需要的流动性列或可交易标记

## 2. 使用历史 Top-K 兼容入口

`backtest_topk` 保留原有签名和默认行为，并把参数转换为 `StrategySpec`、`ExecutionModel` 和 `BacktestSpec` 后调用同一条执行路径。现阶段该入口不会发出弃用警告。

```python
from cstree.backtesting import backtest_topk
```

## 3. 先构造持仓，再单独回测

使用 `StrategySpec` 和 `construct_positions_from_strategy` 生成标准持仓，再把结果交给 `run_position_backtest`。这种方式便于检查目标持仓、保存中间结果和复用定价逻辑。

```python
from cstree.backtesting import StrategySpec, construct_positions_from_strategy
```

## 4. 回放已有目标持仓

使用 `PositionBacktestConfig` 和 `run_position_backtest`。这种方式适合从其他模型、优化器或人工流程接收持仓。

具体用法见根目录 README 的快速开始示例。
