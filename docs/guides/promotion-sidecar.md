# 晋级证据 sidecar

`portfolio_backtester.promotion_sidecar` 根据已构造的目标持仓和历史行情，模拟成交、订单、持仓、现金与约束事件。它用于研究晋级阶段的可交易性证据，不替代执行引擎的真实下单和风控流程。

```python
from portfolio_backtester.promotion_sidecar import (
    PromotionSidecarConfig,
    simulate_promotion_sidecar,
)

result = simulate_promotion_sidecar(
    positions,
    pricing,
    PromotionSidecarConfig(enabled=True),
)
```

输出中的 `events`、`orders`、`fills`、`positions`、`cash` 和 `violations` 可以由上层编排仓保存为运行产物。
