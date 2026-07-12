# 成本与执行假设

本页说明仓库当前支持的成本模型、滑点模型、价格列、流动性约束和退出价格规则。

这些功能用于研究回测。参数需要根据市场、账户、券商和成交方式重新校准，默认值只能作为起点。

## 执行模型的组成

`ExecutionModel` 由五部分组成：

| 组成部分 | 作用 |
| --- | --- |
| `entry_policy` | 指定开仓使用的价格列 |
| `exit_policy` | 指定退出价格列和缺失价格的处理方式 |
| `cost_model` | 估算佣金、税费和其他显式成本 |
| `slippage_model` | 估算买卖价差和市场冲击 |
| `selection_constraints` | 按价格和流动性过滤候选证券 |

执行模型还可以指定交易日历，以及额外的开市日和休市日。

## 成本模型

### 固定基点成本

`BpsCostModel` 按换手率和基点数计算成本。

初始建仓按组合总暴露收取单边成本。后续调仓默认按双边成本处理，可以通过 `round_trip=False` 改为单边。

### 分方向基点成本

`SideBpsCostModel` 可以分别设置：

- 多头开仓成本
- 多头退出成本
- 空头开仓成本
- 空头退出成本
- 空头每日借券成本

这种方式适合不同交易方向具有不同费率的研究。

### A 股明细费用模型

`DetailedTradeFeeModel` 把佣金、印花税、过户费、最低佣金和滑点放在同一个模型中。

直接构造 `DetailedTradeFeeModel()` 时，默认值如下：

| 参数 | 默认值 |
| --- | ---: |
| 买入佣金 | 2.5 个基点 |
| 卖出佣金 | 2.5 个基点 |
| 卖出印花税 | 5.0 个基点 |
| 过户费 | 0.1 个基点 |
| 单笔最低佣金 | 5 元 |
| 买入滑点 | 6.0 个基点 |
| 卖出滑点 | 8.0 个基点 |
| 组合规模 | 1,000,000 元 |

最低佣金需要名义成交金额。模型使用 `portfolio_value` 把权重换手换算为成交金额，因此组合规模会影响成本结果。

通过配置字典构造明细费用模型时，建议显式传入 `buy_slippage_bps` 和 `sell_slippage_bps`，避免依赖构造路径中的默认值。

默认费率不代表任何券商的实时收费标准。使用前应根据账户和回测时期调整。

### 关闭显式成本

`NoCostModel` 返回零显式成本。配置中的 `none`、`off` 和 `zero` 会构造该模型。

## 滑点模型

### 固定基点滑点

`BpsSlippageModel` 按交易权重绝对值乘以固定基点数。

### 参与率滑点

`ParticipationSlippageModel` 使用组合规模、交易权重和流动性列估算成交参与率。

近似计算过程如下：

```text
成交金额 = abs(交易权重) × portfolio_value
参与率 = 成交金额 ÷ amount_col
单证券滑点基点 = base_bps + impact_bps × 参与率 ^ power
```

`max_participation` 可以限制参与率上限。该限制用于稳定估算，不会自动完成分日成交或拒绝超量订单。

流动性列由 `amount_col` 指定。使用开盘价成交时，建议传入开盘前已经可知的滞后流动性指标，例如上一交易日计算完成的 `adv20_amount`。直接使用当日总成交额会引入未来信息。

### 价格分档滑点辅助函数

`l2_price_tiered_slippage` 根据收盘价区间返回一个研究用滑点基点数。卖出方向会在买入基准上增加 2 个基点。

该函数基于仓库内置的价格分档表。它没有读取实时盘口，也没有根据证券、日期和订单规模动态更新。使用者应把它视为简化参数，不应视为真实成交报价。

## 开仓和退出价格

`EntryPolicy` 只负责指定开仓价格列，例如 `open`、`close` 或调用方准备的其他列。

`ExitPolicy` 支持三种退出规则：

| 规则 | 行为 |
| --- | --- |
| `strict` | 计划退出日缺少有效价格或无法交易时，放弃该证券的退出价格 |
| `ffill` | 在计划退出日及之前寻找最近的有效价格 |
| `delay` | 从计划退出日开始向后寻找首个有效价格 |

`delay` 可以配合 `fallback_policy='ffill'`。向后找不到有效价格时，模型会回到计划退出日及之前的最近价格。设置为 `none` 时不会回退。

可交易标记通过调用方指定的布尔列传入。它只能表达该列提供的状态，无法自动补全涨跌停、T+1、停牌原因、订单拒绝和券商规则。

## 价格列和盘中数据

`PositionBacktestConfig` 支持分别设置：

- `price_col`
- `entry_price_col`
- `exit_price_col`

`entry_price_col` 或 `exit_price_col` 为空时，会回退到 `price_col`。

`run_position_backtest` 还可以接收 `intraday_bars`。传入盘中数据后，函数会计算盘中成交量加权价格，并在有结果时替换对应的日线开仓价和退出价。缺失部分继续使用日线价格表。

## `tr_close` 的含义

本包把 `tr_close` 视为调用方提供的普通价格列。仓库不会下载复权因子，也不会构造现金分红账本。

使用 `tr_close` 前，需要由数据提供方明确以下内容：

- 前复权、后复权或总回报口径
- 分红和拆股的处理方式
- 缺失复权因子的回退规则
- 不同证券和时期是否使用一致口径

`tr_close` 适合减少除权除息造成的价格跳变。它无法表示实际分红到账日、税后现金、再投资时点和账户级现金流。

## 配置示例

下面的配置使用分方向费用、参与率滑点、开盘建仓和延迟退出：

```python
from cstree.backtesting.execution import build_execution_model

execution = build_execution_model(
    {
        'cost': {
            'name': 'side_bps',
            'buy_bps': 6,
            'sell_bps': 8,
        },
        'slippage': {
            'name': 'participation',
            'base_bps': 2,
            'impact_bps': 10,
            'amount_col': 'adv20_amount',
            'portfolio_value': 1_000_000,
            'power': 0.5,
        },
        'entry': {
            'price_col': 'open',
        },
        'exit': {
            'price': 'delay',
            'fallback': 'ffill',
            'price_col': 'close',
        },
        'constraints': {
            'min_price': 2,
            'min_amount': 5_000_000,
            'amount_col': 'adv20_amount',
        },
    },
    default_cost_bps=0,
    default_exit_price_policy='strict',
    default_exit_fallback_policy='ffill',
)
```

该配置需要定价数据提供 `open`、`close` 和 `adv20_amount`。

## 适用边界

当前实现适合：

- 日线或低频组合研究
- 比较不同成本和滑点假设
- 检查流动性筛选和退出延迟的敏感性
- 回放外部生成的目标持仓

需要更细实现的场景包括：

- 逐笔或盘口级撮合
- 真实订单队列和部分成交
- 账户级现金、税费和分红账本
- T+1 可卖数量
- 融券可用量和动态借券费
- 券商拒单和交易所微观规则

回测结果高度依赖输入数据和执行假设。报告中应保存价格列、成本参数、滑点参数、组合规模和可交易标记来源。
