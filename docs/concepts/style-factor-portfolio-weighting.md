# 风格因子组合权重

`build_quantile_portfolio_returns` 支持形成日等权和显式价值权重两种初始配置。两种模式都沿用固定份额持有语义：权重只在形成日初始化，之后随价格自然漂移，直到下一次调仓。

## 等权

默认 `weighting="equal"`，每个分组内证券在形成日按 `1 / N` 初始化。该默认值保持历史行为不变。

```python
results = build_quantile_portfolio_returns(
    factors,
    daily,
    rebalance_dates,
    {"size": "factor_size_z"},
)
```

## 价值权重

`weighting="value"` 要求调用方显式提供 `weight_column`。本仓库不假设市值字段名，也不绑定具体市场。

```python
results = build_quantile_portfolio_returns(
    factors,
    daily,
    rebalance_dates,
    {"size": "factor_size_z"},
    weighting="value",
    weight_column="market_cap",
)
```

这里的 `market_cap` 只是调用方字段示例。对每个实际进入分组或 eligible-universe benchmark 的证券，权重必须是有限正数。缺失、NaN、无穷、零或负值都会失败关闭，不会删除异常证券后重新归一，也不会自动退回等权。

## 固定份额语义

价值权重只在形成日按

```text
weight_i = value_i / sum(value)
```

初始化。随后组合保持固定份额，因此强势证券的资本权重会上升，弱势证券会下降。系统不会在持有期内每日重新按 `weight_column` 调整。

当 `include_universe=True` 时，eligible-universe benchmark 与因子分组使用相同的 `weighting` 模式，保证 `long_excess` 的比较口径一致。

这项能力只负责通用组合权重。股票池、微盘定义、市值口径和研究假设由调用方负责。
