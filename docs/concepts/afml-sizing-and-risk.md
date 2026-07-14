# AFML 仓位、HRP 与策略风险

本页说明组合层新增的校准仓位、active-bet 平滑、HRP 和策略风险指标。

## 校准仓位

`portfolio_backtester.bet_sizing` 接受研究层已经严格样本外校准的概率或置信度，并在组合层完成：

- probability-to-size
- 波动率缩放
- 单标的上限
- 仓位离散化
- 最小交易权重
- sizing receipt

```python
from portfolio_backtester import SizingConfig, build_sized_weights

weights = build_sized_weights(
    candidates,
    score_col="signal_backtest",
    config=SizingConfig(
        method="probability_vol_target",
        single_name_cap=0.05,
        step_size=0.005,
        min_trade_weight=0.005,
    ),
)
```

概率校准由 `alpha-research` 负责。组合层不会用训练内概率重新拟合校准器。

## Active bets

`average_active_bets` 根据 `label_start`、`label_end` 和 `bet_size` 平均仍然有效的事件。它适合事件策略和 meta-label 路线，用于降低新预测覆盖旧预测造成的抖动。

`discretize_weights` 将连续目标仓位映射到固定步长。离散化之后仍需重新归一化和执行换手限制。

## HRP

`hierarchical_risk_parity` 和 `rolling_hrp_weights` 面向资产、模型或 sleeve 收益序列。推荐优先用于：

- 多模型信号分配
- value / quality / momentum 等 sleeve 分配
- 多持有期策略分配

HRP 输入必须严格早于调仓日。`rolling_hrp_weights` 使用 `returns.index < rebalance_date`，不会把调仓日收益放入协方差估计。

不建议默认对每日变化的 Top-K 个股直接运行 HRP。动态股票集合和短历史会导致聚类与权重不稳定。个股层 HRP 应额外提供 cluster stability、换手和样本外风险贡献证据。

## 策略风险

`portfolio_backtester.strategy_risk` 提供：

- Probabilistic Sharpe Ratio
- 正收益、负收益和时间集中度 HHI
- hit ratio、average hit、average miss
- implied precision
- strategy failure probability
- implementation shortfall
- shortfall per turnover
- return on execution costs
- cost break-even multiple

`strategy_failure_probability` 衡量在给定盈亏分布和交易频率下，未来评估窗口的 precision 低于目标 Sharpe 所需 precision 的 bootstrap 概率。它描述策略机制失效风险，不能替代账户或组合 VaR。

## 产物

建议 `strategy-pipeline` 保存：

```text
sizing_receipt.json
strategy_risk_report.json
hrp_receipt.json
```

这些文件进入 lineage sidecar，但不进入执行引擎的下单字段。执行输入仍然是标准 `targets.json`。
