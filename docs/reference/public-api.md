# 公开入口

下面这些对象可以直接从 `cstree.backtesting` 导入：

| 类别 | 入口 |
|------|------|
| 分数驱动回测 | `BacktestSpec`、`run_backtest`、`backtest_topk` |
| 策略和持仓构造 | `StrategySpec`、`GroupCap`、`strategy_from_config`、`construct_positions_from_strategy` |
| 持仓回放 | `PositionBacktestConfig`、`PositionBacktestResult`、`run_position_backtest` |
| 持仓契约 | `POSITIONS_BY_REBALANCE_CONTRACT`、`validate_positions_by_rebalance_frame`、`assert_positions_by_rebalance_frame` |
| 成本与滑点 | `DetailedTradeFeeModel`、`l2_price_tiered_slippage` |
| 换手与成本 | `TurnoverBreakdown`、`CostBreakdown`、`name_turnover`、`annualize_turnover`、`turnover_from_trade_weights` |
| 收益汇总 | `summarize_period_returns` |

未列在顶层导出中的模块仍可供仓库内部使用，其接口稳定性低于上表中的公开入口。

完整导出列表见 `src/cstree/backtesting/__init__.py`。
