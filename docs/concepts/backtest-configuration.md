# 回测配置解析

`portfolio_backtester.backtest_config.resolve_backtest_base_settings` 负责解析不依赖数据提供方的通用回测配置。它统一处理以下内容：

- 基准标的和对比基准
- 组合权重、分组限制和选股 tie-break
- 退出方式、退出价格策略和回退策略
- 成本、换手、可交易字段和多空选项
- tearsheet 和回测后处理开关

该函数只返回规范化的回测运行参数。数据字段检查、执行模型构建和执行模拟所需列的补全仍由调用方负责。这样组合回测规则由本仓库维护，pipeline 只负责把数据、评估和运行时配置组合起来。

## 使用方式

```python
from portfolio_backtester.backtest_config import resolve_backtest_base_settings

settings = resolve_backtest_base_settings(
    backtest_cfg,
    eval_top_k=20,
    eval_rebalance_frequency="W",
    eval_transaction_cost_bps=10.0,
    label_horizon_days=5,
)
```

配置错误会通过 `SystemExit` 报告，错误文本保留 `backtest.<option>` 前缀，便于命令行调用方定位配置位置。
