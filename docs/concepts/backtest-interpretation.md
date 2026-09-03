# 回测结果解读

本页说明组合回测侧的结果如何解读。预测质量、IC、CPCV、PBO 和特征重要度的解读见
`alpha-research/docs/concepts/result-interpretation.md`。运行编排、产物目录和 `summary.json`
顶层结构见
[`strategy-pipeline-internal/docs/metrics.md`](https://github.com/runchengxie/strategy-pipeline-internal/blob/main/docs/metrics.md) 与
[`strategy-pipeline-internal/docs/outputs.md`](https://github.com/runchengxie/strategy-pipeline-internal/blob/main/docs/outputs.md)。

## 净收益与毛收益

相关文件：

- `backtest_net.csv`
- `backtest_gross.csv`

优先看净收益。毛收益适合观察未扣成本前的信号表现。净收益反映交易成本和滑点之后还能留下多少。

## 核心统计

`summary.json -> backtest.stats` 常见字段：

- `periods`
- `total_return`
- `ann_return`
- `ann_vol`
- `sharpe`
- `max_drawdown`
- `avg_holding`
- `periods_per_year`
- `avg_turnover`
- `avg_cost_drag`

怎么看：

- `periods`：参与统计的回测周期数。
- `total_return`：整个回测区间累计收益。
- `ann_return`：年化收益。
- `ann_vol`：年化波动。
- `sharpe`：单位风险对应的收益。
- `max_drawdown`：最大回撤。
- `avg_holding`：平均持有时间，通常近似为交易日数。
- `periods_per_year`：年化换算用的周期数。
- `avg_turnover`：每次调仓平均换手。
- `avg_cost_drag`：每期平均被成本拖累多少收益，单位通常是基点。

## 风险指标

额外风险字段：

- `sortino`
- `calmar`
- `drawdown_duration`
- `recovery_time`
- `drawdown_duration_days`
- `recovery_time_days`
- `skew`
- `kurtosis`
- `var_95`
- `cvar_95`

怎么看：

- `sortino`：比 `sharpe` 更关注亏损波动。
- `calmar`：收益相对最大回撤是否划算。
- `drawdown_duration` / `drawdown_duration_days`：从前高跌到低点的周期数或自然日数。
- `recovery_time` / `recovery_time_days`：从低点回到前高的周期数或自然日数。
- `skew`：收益分布偏度。明显为负时，要警惕突发大亏。
- `kurtosis`：收益分布峰度。越高说明极端波动更常见。
- `var_95`：历史最差 5% 情况下的单期收益下限。
- `cvar_95`：落入最差 5% 后的平均损失，通常比 `var_95` 更保守。

## Benchmark 与主动收益

配置 `benchmark_symbol` 或 `benchmark_returns_file` 后，系统会输出：

- `summary.json -> backtest.benchmark`
- `summary.json -> backtest.active`
- `backtest_benchmark.csv`
- `backtest_active.csv`

主动收益常看字段：

- `tracking_error`
- `information_ratio`
- `beta`
- `alpha`
- `corr`
- `active_total_return`

怎么看：

- `tracking_error`：策略收益和基准收益的偏离波动。
- `information_ratio`：主动收益相对主动风险是否划算。
- `beta`：策略和基准的系统性联动。
- `alpha`：剥离 beta 后的独立超额收益。
- `corr`：策略收益和基准收益的相关性。
- `active_total_return`：复利后相对基准的累计超额收益。

## 风格与行业暴露

如果回测有持仓，且 panel 里有可解析的暴露因子，系统会输出风格与行业暴露分析。暴露字段衡量
组合相对基准在风格因子和行业维度上的偏离。`exposure-screen` 协议用于检查组合是否在某些维度过度
集中。

## 滚动 Sharpe

`summary.json -> backtest.rolling_sharpe` 输出滚动 Sharpe，衡量收益稳定性随时间的变化。滚动
研究统计的实现在 `alpha_research.recency_diagnostics`，本页只解释回测产物侧的含义。

## 常见误读

- 分桶 IC、暴露分析和容量压力测试各看一件事，不要混在一起下结论。
- `hit_rate` 是辅助指标，不能替代收益和风险判断。
