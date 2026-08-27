# 通用多袖组合构造

`portfolio_backtester.sleeve_portfolio` 负责把上游已经打分的候选转换为目标持仓。它只拥有通用组合机制，不保存具体策略名称、研究假设或模型版本。

当前公开对象：

```python
from portfolio_backtester.sleeve_portfolio import (
    QuotaSleeveSpec,
    RankBufferedSleeveSpec,
    SleevePortfolioSpec,
    build_sleeve_positions,
    compute_position_changes,
    compute_position_exposure,
)
```

`QuotaSleeveSpec` 适用于按主题、行业或其他分组配额构造的 sleeve。`RankBufferedSleeveSpec` 适用于带入选/退出排名、组内上限和每日替换限制的 sleeve。`SleevePortfolioSpec` 只组合这些机制以及 overlap/weight 规则。

具体策略应在上游 strategy owner 中冻结自己的参数，再显式转换成这些通用 spec。这里不得加入 StyleReplica、DailyWatch20 或其他策略身份常量。

输出遵守 `positions_by_rebalance` 的核心字段约定：`rebalance_date`、`entry_date`、`symbol`、`weight`、`side`，并附带 `leg`、`signal`、`rank` 以及输入 spec 使用的 score/group 描述列。

`compute_position_changes` 和 `compute_position_exposure` 只处理已经构造出的持仓，不重新计算 alpha。执行容量、订单/成交和每日账本继续由 `execution_sim` / position replay 负责。
