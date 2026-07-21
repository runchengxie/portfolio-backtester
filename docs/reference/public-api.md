# 公开入口

下面这些对象可以直接从 `portfolio_backtester` 导入：

| 类别 | 入口 |
|------|------|
| 分数驱动回测 | `BacktestSpec`、`run_backtest`、`backtest_topk` |
| 策略和持仓构造 | `StrategySpec`、`GroupCap`、`strategy_from_config`、`construct_positions_from_strategy` |
| DailyWatch20 兼容入口 | `DailyWatch20Config`、`DailyWatch20Receipt`、`DailyWatch20Result`、`DailyWatch20SelectionError`、`GuardFactorSpec`、`select_daily_watch20` |
| DailyWatch20 组合策略 | `PORTFOLIO_POLICY_SCHEMA`、`DailyWatch20PortfolioPolicy` |
| 旧仓再资格组合 | `INCUMBENT_REQUALIFICATION_SCHEMA`、`IncumbentRequalificationPolicy`、`IncumbentRequalificationConfig`、`IncumbentRequalificationResult`、`IncumbentRequalificationReceipt`、`select_incumbent_requalified_portfolio` |
| 错位持有执行 | `StaggeredCohortExecutionConfig`、`StaggeredCohortExecutionResult`、`simulate_staggered_cohort_execution` |
| 错位持有汇总 | `EXECUTION_SUMMARY_SCHEMA`、`summarize_staggered_execution`、`execution_summary_frame` |
| 持仓回放 | `PositionBacktestConfig`、`PositionBacktestResult`、`run_position_backtest` |
| 持仓基准评估 | `PositionBacktestEvaluation`、`evaluate_position_backtest` |
| 持仓契约 | `POSITIONS_BY_REBALANCE_CONTRACT`、`PositionsByRebalanceFrameContract`、`validate_positions_by_rebalance_frame`、`assert_positions_by_rebalance_frame` |
| 成本与滑点 | `DetailedTradeFeeModel`、`l2_price_tiered_slippage` |
| 交易会话调仓 | `SessionRebalanceSchedule`、`get_session_interval_rebalance_dates` |
| 换手与成本 | `TurnoverBreakdown`、`RebalanceTurnoverReport`、`CostBreakdown`、`name_turnover`、`annualize_turnover`、`turnover_from_trade_weights`、`build_rebalance_turnover_report` |
| 收益汇总 | `summarize_period_returns` |
| 夏普推断 | `probabilistic_sharpe_ratio`、`probabilistic_sharpe_ratio_from_stats`、`deflated_sharpe_ratio`、`expected_max_sharpe`、`sharpe_standard_error`、`annualized_sharpe_to_periodic`、`annualized_variance_to_periodic` |
| 仓位缩放 | `SizingConfig`、`average_active_bets`、`build_sized_weights`、`build_sizing_receipt`、`discretize_weights`、`probability_to_size` |
| 分层风险平价 | `HrpConfig`、`HrpResult`、`hierarchical_risk_parity`、`rolling_hrp_weights` |
| 策略风险 | `StrategyRiskReport`、`implementation_shortfall_metrics`、`return_concentration`、`strategy_failure_probability`、`summarize_strategy_risk` |
| 证据回执 | `build_portfolio_sizing_receipt`、`series_sha256`、`sha256_file`、`write_receipt` |

`probabilistic_sharpe_ratio` 接收收益序列。`probabilistic_sharpe_ratio_from_stats` 接收已经计算好的周期 Sharpe、偏度和超额峰度。

`DailyWatch20` 是现有调用方使用的兼容例外。新增研究假设、特征和晋升规则由研究层与编排层维护。

错位持有执行按 `horizon_days` 建立同样数量的独立 cohort，每个 cohort 初始分配
`1 / horizon_days` 的组合资金。H1 只有一个 cohort，因此占用全部初始资金。汇总中的
`total_return` 是整个 ledger 的累计收益，不是单个 cohort 收益再次除以持有期。

执行容量与每日净值模拟从 `portfolio_backtester.execution_sim` 导入，详细入口见 [执行容量与每日净值模拟](../guides/execution-simulation.md)。AFML 仓位和风险入口见 [AFML 仓位、HRP 与策略风险](../concepts/afml-sizing-and-risk.md)。

未列在顶层导出中的模块仍可供仓库内部使用，其接口稳定性低于上表中的公开入口。

完整导出列表见 `src/portfolio_backtester/__init__.py`。
