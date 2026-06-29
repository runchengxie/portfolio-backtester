# 持仓和快照输出

> status: active
> owner: portfolio-backtester
> last_verified: 2026-06-29
> source_of_truth: yes
> superseded_by: n/a

迁移说明：本页从 `strategy-pipeline/docs/reference/outputs/positions.md` 迁入。组合持仓和回测持仓字段由 `portfolio-backtester` 维护；live 目标导出仍由 `strategy-pipeline` 维护。

| 文件 | 用途 |
| --- | --- |
| `positions_current.csv` | 最新目标理论持仓 |
| `positions_by_rebalance.csv` | 每个调仓日的历史目标持仓 |
| `trades_by_rebalance.csv` | 调仓差异和交易近似 |
| `latest.json` | live 输出场景的便利指针 |

`latest.json` 是 mutable 指针；发布、复现和归档使用具体 run 目录。完整字段参考暂时仍在 `strategy-pipeline/docs/reference/outputs/full-reference.md`。
