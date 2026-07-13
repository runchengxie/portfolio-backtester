# 组合式回测规范

`BacktestSpec` 是分数驱动回测的推荐配置入口。它复用现有的 `StrategySpec` 和 `ExecutionModel`，把原来散落在 `backtest_topk` 参数中的设置组合成一个不可变对象。

## 职责划分

| 对象 | 负责内容 |
| --- | --- |
| `StrategySpec` | 分数列、Top-K 数量、多空模式、权重方法、持仓缓冲和分组数量上限 |
| `ExecutionModel` | 开仓价、退出规则、成本、滑点、交易日历和价格或流动性筛选约束 |
| `BacktestSpec` | 策略与执行模型的组合、调仓日期、持有期、年化口径和其他运行设置 |

`BacktestSpec` 不定义新的选择器、权重分配器、成本模型或退出规则类型。策略和执行语义继续由仓库中已有的类型负责。

## 基础示例

下面省略 `scores` 的 `DataFrame` 构造。输入至少需要 `trade_date`、`symbol`、分数列和执行模型使用的价格列。

```python
import pandas as pd

from cstree.backtesting import BacktestSpec, StrategySpec, run_backtest
from cstree.backtesting.execution import build_execution_model

execution = build_execution_model(
    {
        'cost': {'name': 'bps', 'bps': 10},
        'entry': {'price_col': 'close'},
        'exit': {
            'price': 'strict',
            'fallback': 'ffill',
            'price_col': 'close',
        },
    },
    default_cost_bps=0,
    default_exit_price_policy='strict',
    default_exit_fallback_policy='ffill',
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

`run_backtest` 的返回值与 `backtest_topk` 保持一致。没有可计算的持有期时返回 `None`。有结果时返回统计字典、净收益序列、毛收益序列、换手率序列和持有期明细。

## 配置序列化

`BacktestSpec` 是 `frozen=True` 的数据类。`to_mapping()` 会把调仓日期和内置执行组件转换为适合 JSON 或 YAML 的值：

```python
import json

payload = spec.to_mapping()
encoded = json.dumps(payload, ensure_ascii=False)
restored = BacktestSpec.from_mapping(json.loads(encoded))

assert restored == spec
```

映射包含 `schema_version`。当前版本为 1，读取未知版本时会直接报错，避免静默采用错误语义。

信号表和行情表不属于配置，因此不会写入映射。信号与定价使用同一个表时调用：

```python
run_backtest(scores, spec)
```

筛选后的信号表缺少完整退出价格时，可以另传只读定价表：

```python
run_backtest(filtered_scores, spec, pricing_data=published_prices)
```

## 历史入口兼容

`backtest_topk` 继续保留原有参数、默认值、返回结构和异常行为。兼容入口会完成以下映射：

| 历史参数 | 新对象中的位置 |
| --- | --- |
| `pred_col`、`top_k`、`weighting`、`long_only` | `StrategySpec` |
| `buffer_exit`、`buffer_entry` | `StrategySpec` |
| `group_col`、`max_names_per_group` | `StrategySpec.group_cap` |
| `price_col`、`cost_bps`、退出规则 | 默认 `ExecutionModel` |
| 显式 `execution` | `BacktestSpec.execution` |
| 调仓、持有期、流动性和排序设置 | `BacktestSpec` |

显式传入 `execution` 时，它仍覆盖 `price_col`、`cost_bps` 和历史退出参数，与原有行为一致。兼容入口暂不发出弃用警告，下游调用完成审计后再决定迁移期限。

## 适用边界

`BacktestSpec` 描述分数驱动的 Top-K 组合回测。已有目标持仓的确定性回放继续使用 `PositionBacktestConfig` 和 `run_position_backtest`。配置对象不负责数据下载、模型训练、任务编排或实盘执行。
