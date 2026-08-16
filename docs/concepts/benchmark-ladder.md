# 市场 benchmark 阶梯

本页定义组合回测侧的市场 benchmark 选择与报告层对比协议。特征研究协议、晋升门槛和模型分层见
`alpha-research/docs/concepts/feature-research-protocol.md` 与
`alpha-research/docs/concepts/overfitting-controls.md`。运行编排和配置入口见
`strategy-pipeline/docs/concepts/benchmark-protocol.md`。

## 市场 benchmark

HK 研究默认用 `backtest.benchmark_symbol: 02800.HK`。如果要评估相对港股通可投 universe 的
alpha，更贴近的做法是自建一个港股通 by-date universe cap-weight benchmark，然后通过
`backtest.benchmark_returns_file` 接入同一条回测，避免强行用单一 ETF 近似。

配置后，run 会额外输出：

- `summary.json -> backtest.benchmark`
- `summary.json -> backtest.active`
- `backtest_benchmark.csv`
- `backtest_active.csv`

如果想保留市场 ETF 代理和更贴近 universe 的 alpha benchmark，可以把主 benchmark 固定在其中一条，
再把其他对照放进 `backtest.benchmark_compare`。这层是报告层附加对比，不会改变主 benchmark 的口径。
配置后还会额外输出：

- `summary.json -> backtest.report_file`
- `summary.json -> backtest.tearsheet_file`
- `summary.json -> backtest.benchmark_compare`
- `backtest_report.csv`
- `backtest_tearsheet.html`
- `backtest_benchmark_compare_summary.csv`
- `backtest_benchmark_compare_<name>.csv`

## 报告层 benchmark 阶梯

如果要在不同 benchmark 文件之间做正式报告层比较，可以用独立 ladder：

```bash
strategy backtest benchmark-ladder \
  --config strategy-pipeline/configs/experiments/sweeps/a_share__research_protocol_benchmark_ladder.yml
```

这层不改变单次 run 的主 benchmark，只读取已有策略收益和 benchmark 收益，输出每条 benchmark 的
active total return、IR、tracking error、beta、alpha、相关性、可比状态和 attribution 文件可用性。

## HK selected 默认 benchmark 阶梯

历史 HK selected 研究把 benchmark protocol 定成下面这套：

| 层级 | 角色 |
| --- | --- |
| 市场 benchmark | 回测市场对照，默认 `02800.HK`，月频本地推荐用 universe-aligned `benchmark_returns_file` |
| 报告层 compare benchmark | 同一条 run 内并排看 ETF / selected cap-weight / connect cap-weight |
| 特征 benchmark 1 | 季度纯量价 floor |
| 特征 benchmark 2 | 季度 core PIT 增量 |
| 强 benchmark | 季度 core PIT + 慢量价，默认要被超越的对象 |
| 线性 benchmark | 同一 hybrid 单元上的 sanity check |
| Challenger | 同一 hybrid 单元上的排序模型或稀疏线性模型 |

历史归档配置文件名与复现入口见 `strategy-pipeline/docs/archive/research/hk/README.md`。

这套协议先把问题拆开：

1. alpha 是不是先出现在纯量价里
2. 加 core PIT 后有没有稳定增量
3. 再加慢量价后有没有继续增量
4. 在同一条 hybrid 路线上，模型差异到底带来了什么
