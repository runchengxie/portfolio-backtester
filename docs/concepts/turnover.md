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

## 年化

`annualize_turnover` 只做线性年化，用于描述交易强度，不代表可复利收益。
