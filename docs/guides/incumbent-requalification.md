# 旧仓再资格组合构造

`portfolio_backtester.incumbent_requalification` 提供一个与模型无关的组合构造接口，适合候选池变化速度明显快于经济信号的策略。

这套政策把两个容易混在一起的决策分开处理：

- 新仓准入：新标的必须满足当前严格候选条件，并进入 `entry_rank_limit`。
- 旧仓退出：已有持仓会使用当日信息重新评分。只要仍满足硬资格，并处于更宽的 `exit_rank_limit` 内，就可以继续持有。

候选池成员变化不会直接成为无条件卖出信号。旧仓也不能无限延续，它必须具备当日评分、当日硬资格，并处于冻结的退出缓冲区内。

## 最小示例

```python
import pandas as pd

from portfolio_backtester import (
    IncumbentRequalificationPolicy,
    select_incumbent_requalified_portfolio,
)

candidates = pd.DataFrame(
    {
        "trade_date": ["2026-07-20"] * 4,
        "symbol": ["A", "B", "C", "D"],
        "selection_score": [0.90, 0.80, 0.70, 0.60],
        "industry": ["tech", "tech", "health", "industrial"],
        "hard_eligible": [True, True, True, True],
        "entry_eligible": [True, True, False, True],
    }
)

result = select_incumbent_requalified_portfolio(
    candidates,
    previous_symbols=["C"],
    policy=IncumbentRequalificationPolicy(
        portfolio_size=3,
        entry_rank_limit=3,
        exit_rank_limit=4,
        max_new_positions=1,
        industry_cap=2,
    ),
)

positions = result.positions
receipt = result.receipt.to_dict()
```

`C` 虽然不具备新仓准入资格，但它是退出缓冲区内的旧仓，因此可以继续持有。`entry_eligible=False` 的标的永远不能作为新仓进入组合。

## 现金语义

每个入选标的占用一个固定组合槽位：

```text
target_weight = 1 / portfolio_size
```

当硬退出数量超过每日新增预算时，空缺槽位保留为现金。系统不会把剩余持仓重新归一到 100% 总敞口，避免把弱信号或缺失信号伪装成更高的集中度。

设置 `allow_cash=False` 后，只要冻结政策无法填满组合，构造过程就会失败关闭。

## 替换语义

正常调仓最多新增 `max_new_positions` 只标的。首次建仓允许直接填充至 `portfolio_size`。

组合已经满仓时，新候选优先挑战已经跌出准入区、但仍处于退出缓冲区的旧仓，然后再挑战准入区内较弱的旧仓。替换还必须同时满足：

- 分数改善达到 `min_score_improvement`
- 替换后的行业持仓不超过 `industry_cap`

## 默认输入字段

| 字段 | 含义 |
| --- | --- |
| `trade_date` | 唯一的组合决策日期 |
| `symbol` | 唯一证券代码 |
| `selection_score` | 当日可比较的排序分数 |
| `industry` | 用于组合上限的 PIT 安全行业标签 |
| `hard_eligible` | 当日交易和安全硬资格 |
| `entry_eligible` | 新开仓使用的严格准入资格 |

调用方字段名称不同时，通过 `IncumbentRequalificationConfig` 显式映射。

## 证据边界

这个接口只生成目标持仓和可审计回执。它不能证明策略收益改善，也不模拟 T+1、部分成交、涨跌停排队或券商拒单。候选政策进入晋级判断前，仍需与基线使用相同的冻结执行器、成本模型和样本外协议完成比较。
