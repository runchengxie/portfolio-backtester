# 执行容量与每日净值模拟

`portfolio_backtester.execution_sim` 提供容量成交、执行后每日净值和理想每日净值三类模拟。该子包具有独立公开入口，当前没有从包根重新导出。

## 入口

```python
from portfolio_backtester.execution_sim import (
    ExecutionSimConfig,
    simulate_capacity_execution,
    simulate_execution_adjusted_nav,
    simulate_ideal_daily_nav,
)
```

- `simulate_capacity_execution` 输出订单、成交和汇总
- `simulate_execution_adjusted_nav` 输出每日净值、订单、成交和汇总
- `simulate_ideal_daily_nav` 假设目标仓位立即完成，用作充分流动性对照

## 配置

`ExecutionSimConfig` 默认关闭。启用后的主要默认值如下：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `portfolio_value` | 1,000,000 | 组合名义规模 |
| `participation_rate` | 0.05 | 单日成交参与率 |
| `liquidity_cols` | `medadv20_amount`、`amount` | 容量约束使用的流动性列 |
| `buy_max_days` | 5 | 买单最长等待天数 |
| `sell_max_days` | 10 | 卖单最长等待天数 |
| `zero_fill_abort_days_buy` | 5 | 连续零成交后的买单终止天数 |
| `unfilled_buy_action` | `keep_cash` | 未成交买单保留现金 |
| `unfilled_sell_action` | `keep_position` | 未成交卖单保留持仓 |

`build_execution_sim_config` 负责读取配置映射，`required_execution_sim_columns` 返回启用模拟后需要的价格和流动性列。

## 输入与结果

目标持仓至少需要调仓日、建仓日、证券代码和权重。当前模拟只处理多头正权重。行情表需要交易日期、证券代码、价格列和配置中的流动性列。买卖方向可分别传入可交易标记。

`ExecutionSimResult` 包含 `summary`、`orders` 和 `fills`。`ExecutionAdjustedNavResult` 额外包含 `daily`，其中记录每日净值、现金和敞口等结果。

其余公开对象包括 `SELL_UNTIL_NEXT_REBALANCE`、`TradeFeeModel`、`describe_execution_sim_config` 和 `describe_trade_fee_model`。它们分别用于延迟卖出期限、费用协议以及配置和费用说明的序列化。

容量模拟根据参与率限制成交，并保留未成交余量。它不会自动补全涨跌停、T+1、整手、停牌原因或券商拒单规则。调用方应通过规范化行情和可交易字段提供所需约束。

## 使用边界

理想每日净值用于比较充分流动性情形。执行后每日净值用于观察延迟成交、未成交和成本拖累。两者都属于研究模拟，不能替代真实订单状态、账户现金账本和券商风控。
