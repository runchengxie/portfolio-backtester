# 回测输出契约

本页记录 portfolio 组合回测侧输出的报告文件与字段契约。运行编排、产物目录和 `summary.json`
顶层结构见
[`strategy-pipeline-internal/docs/outputs.md`](https://github.com/runchengxie/strategy-pipeline-internal/blob/main/docs/outputs.md) 与
[`strategy-pipeline-internal/docs/reference/outputs/full-reference.md`](https://github.com/runchengxie/strategy-pipeline-internal/blob/main/docs/reference/outputs/full-reference.md)。

## 固定分数组合层对比

`strategy backtest construction-grid` 默认不写文件。如果传 `--output` 或 `output_csv`，会写
CSV。如果传 `--output-json`，会写 JSON。

来源：

1. 读取 construction-grid YAML。
1. 读取指定 run 的 `summary.json` 和既有 `eval_scored.parquet`。
1. 在同一份固定 score 上评估多组组合构建参数，不重训模型。

列契约（CSV 与 JSON 行字段一致）：

```text
variant,scored_file,summary_path,target_col,price_col,eval_signal_col,backtest_signal_col,top_k,short_k,long_only,cost_bps,buffer_exit,buffer_entry,weighting,weighting_liquidity_col,liquidity_floor_col,liquidity_floor_quantile,max_turnover_per_rebalance,score_postprocess_method,score_postprocess_columns,dynamic_ensemble_active,dynamic_ensemble_signal_cols,dynamic_ensemble_avg_active_factor_count,dynamic_ensemble_avg_factor_turnover,dynamic_ensemble_avg_stock_turnover,factor_correlation_threshold,risk_penalty_columns,risk_penalty_strength,eval_ic_mean,eval_ic_ir,eval_long_short,eval_turnover_mean,backtest_periods,backtest_total_return,backtest_gross_total_return,backtest_ann_return,backtest_ann_vol,backtest_sharpe,backtest_max_drawdown,backtest_avg_turnover,backtest_avg_cost_drag,active_total_return,information_ratio,tracking_error,beta,alpha,corr,benchmark_name,benchmark_returns_file,exposure_available,status,error
```

补充：

1. `status=ok` 表示该 variant 成功形成组合并完成回测。
1. `status=failed` 表示该 variant 缺少 scored artifact、列、benchmark 或无法形成组合。`error`
   给出原因。
1. `backtest_total_return` 是扣除成本后的净收益。`backtest_gross_total_return` 是同一持仓路径下
   未扣成本的收益。
1. `active_*`、`information_ratio`、`tracking_error`、`beta`、`alpha`、`corr` 只在配置了可用
   benchmark returns 时有值。
1. `exposure_available` 仅表示该 variant 有可用于后续暴露或归因检查的输入线索。当前不隐式执行
   行业或风格中性化。
1. 如果配置 `rolling_selection.output_json`，会额外写
   `portfolio_backtester.construction_grid_rolling_selection` JSON。该报告按 `objective_col` 排序，
   但会对切换 variant 施加 `switch_penalty + min_improvement` 门槛。未达门槛时保留
   `previous_variant`。

## Benchmark 阶梯输出

`strategy backtest benchmark-ladder` 输出每条 benchmark 的 active total return、IR、tracking
error、beta、alpha、相关性、可比状态和 attribution 文件可用性。协议见
`docs/concepts/benchmark-ladder.md`。

## 持仓输出

`positions_by_rebalance.csv`、`positions_current.csv` 与调仓差异文件字段见
`docs/reference/outputs/positions.md`。

## 执行模拟与容量

`execution_sim_*.csv` 与 `capacity_*.json` 描述执行模拟与容量压力测试产物。相关协议见
`docs/guides/execution-simulation.md` 与 `docs/concepts/afml-sizing-and-risk.md`。
