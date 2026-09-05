# 执行容量与每日净值模拟

`portfolio_backtester.execution_sim` 提供容量成交、执行后每日净值和理想每日净值三类模拟。该子包具有独立公开入口，当前没有从包根重新导出。

## 入口

```python
from portfolio_backtester.execution_sim import (
    ExecutionSimConfig,
    simulate_capacity_execution,
    simulate_execution_adjusted_nav,
    simulate_ideal_daily_nav,
    PreparedExecutionTables,
    to_unified_ledger,
    UnifiedLedger,
)
```

- `simulate_capacity_execution` 输出订单、成交和汇总。`prepare_execution_tables` 预先整理可复用的执行数据表
- `simulate_execution_adjusted_nav` 输出每日净值、订单、成交和汇总
- `simulate_ideal_daily_nav` 假设目标仓位立即完成，用作充分流动性对照
- `to_unified_ledger` 把模拟结果适配为路线图定义的八个统一账本字段，也可通过 `ExecutionSimResult.to_unified_ledger` 与 `ExecutionAdjustedNavResult.to_unified_ledger` 方法调用
- `UnifiedLedger` 是统一账本的数据容器，包含 `targets`、`orders`、`fills`、`daily_positions`、`daily_cash`、`daily_nav`、`cost_breakdown` 和 `turnover_breakdown`

## 配置

`ExecutionSimConfig` 默认关闭。启用后的主要默认值如下：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `portfolio_value` | 1,000,000 | 组合名义规模 |
| `participation_rate` | 0.05 | 单日成交参与率 |
| `liquidity_cols` | `medadv20_amount`、`amount` | 容量约束使用的流动性列 |
| `liquidity_notional_multiplier` | 1.0 | 将流动性列换算为组合名义货币单位的乘数，Tushare `amount` 为千元时应设为 1,000 |
| `buy_max_days` | 5 | 买单最长等待天数 |
| `sell_max_days` | 10 | 卖单最长等待天数 |
| `zero_fill_abort_days_buy` | 5 | 连续零成交后的买单终止天数 |
| `unfilled_buy_action` | `keep_cash` | 未成交买单保留现金 |
| `unfilled_sell_action` | `keep_position` | 未成交卖单保留持仓 |

`build_execution_sim_config` 负责读取配置映射，`required_execution_sim_columns` 返回启用模拟后需要的价格和流动性列。

## 输入与结果

目标持仓至少需要调仓日、建仓日、证券代码和权重。当前模拟只处理多头正权重。行情表需要交易日期、证券代码、价格列和配置中的流动性列。买卖方向可分别传入可交易标记。

`ExecutionSimResult` 包含 `summary`、`orders` 和 `fills`。`ExecutionAdjustedNavResult` 额外包含 `daily`，其中记录每日净值、现金和敞口等结果。

其余公开对象包括 `SELL_UNTIL_NEXT_REBALANCE`、`TradeFeeModel`、`PreparedExecutionTables`、`describe_execution_sim_config` 和 `describe_trade_fee_model`。它们分别用于延迟卖出期限、准备后的执行表、费用协议以及配置和费用说明的序列化。

容量模拟根据参与率限制成交，并保留未成交余量。`round_lot`、`enforce_t1`、`enforce_price_limits` 和 `enforce_listing_status` 可显式启用整手、T+1、价格限制与上市状态检查，默认关闭。调用方仍须提供相应的原始价格、涨跌停价格、上市状态和买卖可交易字段。停牌原因与券商拒单规则不会自动补全。

整手数量与价格限制必须使用一致的真实股份和价格单位。不能直接把复权价格对应的合成份额视为真实可交易股数，也不能把原始涨跌停价格与复权成交价格直接比较。使用长期历史数据时，还需单独验证分红、拆并股等公司行动的处理。

## 使用边界

理想每日净值用于比较充分流动性情形。执行后每日净值用于观察延迟成交、未成交和成本拖累。两者都属于研究模拟，不能替代真实订单状态、账户现金账本和券商风控。
