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

from portfolio_backtester import BacktestSpec, StrategySpec, run_backtest
from portfolio_backtester.execution import build_execution_model

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

## 弱信号、新增名称和固定槽位控制

`BacktestSpec` 提供两个默认关闭的研究参数：

- `selection_min_score` 是硬性分数门槛。多头等降序选择保留分数大于等于门槛的证券，空头等升序选择保留分数小于等于门槛的证券。缺失或无法转换为数值的分数不合格。门槛在持仓缓冲和分数差保留规则之前执行，因此历史持仓也不能越过门槛。
- `max_new_names_per_rebalance` 限制一次调仓中相对上一期非空持仓新增的证券数量。首次非空建仓不受该限制，此前只有空筛选期时仍按初始建仓处理。多空策略会分别计算多头侧和空头侧的新增数量。

两个字段均为 `None` 时完全沿用原有 Top-K 行为。启用后，合格证券或允许新增的证券不足时，选择器不会用较弱证券补满 `top_k`。价格、流动性和可交易性约束先于新增名称计数执行，未通过执行约束的证券不会消耗新增额度。分组数量上限继续作用于最终持仓。

启用任一控制后，某一侧没有合格证券时，Top-K 收益回放会把该侧记为现金：毛收益为零，从已有持仓切到现金仍计算卖出换手与成本，且该期不会从收益序列中删除。持仓明细入口用`该期没有该侧的行`表示现金。首次出现合格证券前的空筛选期不会消耗初始建仓额度。

`selection_min_score` 的资格约束优先于目标权重换手上限。启用 `max_turnover_per_rebalance` 时，插值后的目标权重也会再次剔除低于门槛的旧持仓，因此硬门槛可能使实际权重换手超过上限。

```python
conservative_spec = BacktestSpec(
    strategy=spec.strategy,
    execution=spec.execution,
    rebalance_dates=spec.rebalance_dates,
    shift_days=spec.shift_days,
    trading_days_per_year=spec.trading_days_per_year,
    selection_min_score=0.25,
    max_new_names_per_rebalance=2,
)
```

以下三个字段用于建立不靠候选回填、且能明确保留现金的研究基线：

- `entry_rank_cutoff` 是新证券的严格排名上限。设置为 `8` 时，新证券只有进入前 8 名才能买入。缓冲区允许旧持仓在更宽的退出排名内继续持有，但选择器不会用第 9 名以后的新证券补满组合。
- `target_weight_policy="fixed_slot"` 只支持多头等权组合。每个目标槽位固定为 `1 / top_k`，不足 `top_k` 的部分保留现金。例如 Top10 只选出 8 只时，每只权重为 `0.10`，目标总敞口为 `0.80`。
- `selection_price_policy="target_first"` 先用信号冻结目标名单，再单独检查开仓日价格、流动性和可交易性。未通过开仓约束的目标不由更低排名证券替换，其权重在模型持仓中保留为现金。

三个字段默认分别为 `None`、`"normalized"` 和 `"execution_aware"`，因此旧配置和旧调用结果不变。`fixed_slot` 与非等权或多空策略组合会直接报错。低换手 Top10 基线可以组合使用 `buffer_exit=5`、`buffer_entry=2`、`entry_rank_cutoff=8`、`target_weight_policy="fixed_slot"` 和 `selection_price_policy="target_first"`，具体阈值仍需由独立实验验证。

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
| 调仓、持有期、流动性、排序、分数门槛、新增名称和固定槽位限制 | `BacktestSpec` |

显式传入 `execution` 时，它仍覆盖 `price_col`、`cost_bps` 和历史退出参数，与原有行为一致。兼容入口暂不发出弃用警告，下游调用完成审计后再决定迁移期限。

## 适用边界

`BacktestSpec` 描述分数驱动的 Top-K 组合回测。已有目标持仓的确定性回放继续使用 `PositionBacktestConfig` 和 `run_position_backtest`。配置对象不负责数据下载、模型训练、任务编排或实盘执行。
