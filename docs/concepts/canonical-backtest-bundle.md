# Canonical 回测证据 bundle

`portfolio_backtester.backtest_result.v1` 把一次组合回测的统一账本保存成不可覆盖、可哈希、可重载的目录。
它服务于研究证据和跨仓交接，不替代现有 `CanonicalBacktestResult` 返回对象，也不创建第二套订单、成交或现金模型。

## 证据等级

`BacktestEvidenceTier` 第一版只有两级：

- `diagnostic`：允许理想化成交、目标权重收益或缺少完整订单/现金证据的兼容研究。它可以写 bundle，
  但下游晋级门禁不得把它当成可执行净值证据。
- `execution_aware`：要求 backend capability 明确支持 `order_lifecycle` 和 `daily_ledger`，要求完整
  `research.clock.v1` 执行窗口，并要求统一账本账户对账通过。

`execution_aware` 不要求每次运行一定出现订单或成交行。没有交易的合法区间仍可以拥有空表；关键是
订单/成交表和 daily ledger 能力真实存在，且文件与 capability 不互相撒谎。

## 目录

Writer 固定保存：

```text
backtest_result/
  manifest.json
  targets.parquet
  orders.parquet
  fills.parquet
  daily_positions.parquet
  daily_cash.parquet
  daily_nav.parquet
  cost_breakdown.parquet
  turnover_breakdown.parquet
  diagnostics.json
```

`manifest.json` 保存：

- schema、run id 和 evidence tier；
- `research.artifact-envelope.v2`；
- `research.clock.v1` mapping；
- backend 身份和 capability；
- 上游 artifact 引用；
- 每个文件的 SHA-256、required 状态和行数；
- reconciliation 结果。

Artifact Envelope 的 `content_sha256` 对 canonical inventory 做哈希。manifest 本身不进入 inventory，
避免自引用哈希。

## 账户对账

当前统一账本提供日度聚合的：

```text
daily_positions.positions_value
daily_cash.cash
daily_nav.nav
```

`reconcile_unified_ledger` 按 `trade_date` 一对一对齐，并检查：

```text
nav = cash + positions_value
```

允许相对于账户规模的浮点容差。日期覆盖不一致、重复日期、非数值会计列或超过容差都会直接失败。

## 写入与读取

`write_backtest_bundle` 先在最终目录同级创建临时目录，写完 Parquet/JSON、计算 inventory 哈希并构造
Artifact Envelope，最后通过原子目录替换发布。目标目录已存在时直接拒绝覆盖；任何中间失败都会清理临时目录。

`read_backtest_bundle` 默认验证：

- manifest schema；
- evidence-tier completeness；
- Artifact Envelope；
- inventory content hash；
- inventory 中每个文件的内容 SHA-256。

Reader 不重新执行组合、订单、成交或 PnL 计算。

## ResearchClock 边界

本仓当前已固定依赖 `research-contracts` 的 main 可达提交。该提交尚未包含新的 `ResearchClock` 类型时，
bundle 只保存并检查 `research.clock.v1` 的 execution-aware 必需字段，不复制完整时钟解析器。
等 workspace 契约种子 PR 合入后，下游 producer 升级 `research-contracts` pin，由根 `ResearchRunManifest`
执行完整时钟因果校验。

这样可以让 canonical bundle PR 独立审查，同时避免把功能分支 commit 写进正式 dependency pin。
