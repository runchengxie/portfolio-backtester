# 换手率口径

`portfolio-backtester` 区分两类容易被混用的换手率：

| 指标 | 含义 | 适用场景 |
|------|------|---------|
| `name_turnover` | 持仓名称替换比例 | 描述 Top-K 名单稳定性 |
| `TurnoverBreakdown.one_way_turnover` | 基于目标权重变化的单边换手率 | 成本核算 |

## TurnoverBreakdown 字段

`TurnoverBreakdown` 同时保留以下字段，避免只报告一个含义模糊的 `turnover`：

- `buy_weight`：买入权重合计
- `sell_weight`：卖出权重合计
- `gross_traded_weight`：买卖总成交权重
- `half_l1_turnover`：严格数学定义的双边换手率
- `one_way_turnover`：用于成本核算的单边换手率

对于非初始调仓：

```text
half_l1_turnover = 0.5 * sum(abs(target_weight - drifted_weight))
```

初始建仓沿用历史成本口径，`one_way_turnover` 等于实际买入的总敞口。`half_l1_turnover` 仍保留严格的数学定义。

## 调仓报告分层

`RebalanceTurnoverReport` 分开记录名单目标、权重目标、盘前需求和实际执行：

- `target_entered_names`、`target_exited_names`、`target_overlap_names` 记录进入、退出和重合名单，对应的 `*_count` 字段记录数量
- `target_name_turnover` 记录目标名单替换比例，保留原有字段兼容性
- `target_weight_full_l1` 和 `target_weight_half_l1` 比较前后两期目标权重
- `pretrade_demand_*` 比较价格漂移后的持仓和本次请求权重
- `executed_buy` 和 `executed_sell` 记录实际买卖权重
- `executed_full_l1` 和 `executed_half_l1` 明确记录实际执行换手，`executed_gross` 保留为 `executed_full_l1` 的兼容别名
- `executed_cost` 只在调用方提供实际成交成本时有值
- `target_gross_exposure` 和 `target_cash_weight` 记录冻结目标的总敞口与现金
- `modeled_gross_exposure` 和 `modeled_cash_weight` 记录开仓约束生效后的模型持仓敞口与现金

权重和换手字段都以组合期初净值为单位，`1.0` 表示净值的 100%。`executed_cost` 也使用期初净值的收益拖累单位，不使用货币金额。

分数回测没有成交回报，因此 period 输出中的 `executed_buy`、`executed_sell`、`executed_gross`、`executed_full_l1`、`executed_half_l1` 和 `executed_cost` 为 `None`。`modeled_fee_cost`、`modeled_slippage_cost` 和 `modeled_total_cost` 是模型估计，不冒充实际成交。

汇总结果中的 `avg_*` 包含首次建仓。`avg_rebalance_*` 排除 `is_initial_build=true` 的周期，用于报告持续换手。上述目标和模型敞口字段也提供对应的 `avg_*` 汇总。

固定 Top10 等权组合替换两只股票时，目标权重的 full-L1 为 `0.40`，half-L1 为 `0.20`。`max_positive_names` 可用于拒绝权重插值产生的超额正权重名称。

按交易会话间隔调仓使用 `SessionRebalanceSchedule`。`rebalance_interval_sessions=3` 表示每三个交易会话整仓调仓一次，并持有到下一次调仓。该口径不表示每天启动一个三日 sleeve。

## 年化

`annualize_turnover` 只做线性年化，用于描述交易强度，不代表可复利收益。
