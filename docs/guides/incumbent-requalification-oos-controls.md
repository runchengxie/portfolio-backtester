# 旧仓再资格 OOS 对照桥

`portfolio_backtester.incumbent_requalification_oos` 提供两条共享同一组合政策和诊断口径的逐日 OOS 桥：

- `stateful_incumbent_requalification_daily_rows`：把上一日目标持仓传给选择器，允许旧仓在退出缓冲区内继续持有。
- `stateless_incumbent_requalification_daily_rows`：每日重置选择状态，但仍跨日计算目标权重变化和换手。

两条路径的唯一区别是选择阶段是否携带旧仓状态。研究应用可以用 `stateful - stateless` 隔离持仓缓冲本身的影响，再用生产基准比较整套候选策略。

## 失败关闭

新版桥要求输入显式提供：

- `hard_eligible`：全市场当日硬资格。
- `entry_eligible`：新开仓使用的严格候选资格。

缺少任一字段都会失败。桥不会再把 `entry_eligible` 静默替换成 `hard_eligible`，避免全市场准入被误报为热点候选池实验。

## 自定义列

`IncumbentRequalificationConfig` 可以映射日期、证券、分数、行业和资格字段。桥会在调用共享执行诊断前把日期和证券列转换为 `trade_date` 与 `symbol`，因此自定义映射对选择和回放保持一致。

执行行情 `frame` 仍使用组合回放的标准列：

```text
trade_date
symbol
open
up_limit
down_limit
is_suspended
```

## 研究边界

无状态对照不是生产策略。它用于回答持仓缓冲是否降低换手或改变收益。生产比较仍应另外保留冻结的实际基准，并使用相同评分、候选 membership、成本模型和完整日期集合。
