# 回测后端边界

本页记录 `portfolio-backtester` 当前已经落地的后端边界，以及历史候选、参考框架和规划项的真实状态。

## 当前结论

当前主分支只有 `NativePositionReplayBackend` 实现。当前 registry 只包含 `native.position_replay`。

```text
外部信号或目标持仓
        |
        v
组合选择、权重与约束
        |
        v
NativePositionReplayRequest
        |
        v
NativePositionReplayBackend
        |
        v
CanonicalBacktestResult
```

`BackendRegistry` 提供显式登记和查找机制，不会自动发现插件。主分支没有第三方回测框架的适配器，也没有对应的运行时依赖。

## 框架状态

| 名称 | 当前状态 | 当前用途 |
| --- | --- | --- |
| `native.position_replay` | 已实现并作为权威后端 | 确定性持仓周期回放、A 股规则、成本和容量研究 |
| Qlib | 历史候选没有进入 `main` | 当前无运行职责 |
| LEAN | 历史候选没有进入 `main` | 只保留架构参考用途 |
| Backtrader | 仍处于规划阶段 | 当前没有适配器、依赖或 registry 项 |
| vn.py | 属于本仓库范围外 | Gateway、实时传输和券商执行由执行系统决定 |

Qlib 与 LEAN 的历史候选没有进入 `main`。LEAN 只保留架构参考用途。文档和测试不得把这些历史工作描述成可调用能力。

Backtrader 仍处于规划阶段。任何后续实现都需要独立设计评审、可选依赖、规范化结果转换和固定场景对照证据。

vn.py 属于本仓库范围外。本仓库不维护 Gateway、实时订单传输或券商账户状态。

机器可读状态见 [框架状态账本](../framework-integration-ledger.yml)。账本把当前后端、历史候选、规划项和范围外项目分开记录。

## 稳定边界

### 执行契约

`portfolio_backtester.execution_contracts` 定义框架中立的类型：

- `Instrument`
- `Target`
- `OrderIntent`
- `OrderEvent`
- `Fill`
- `LedgerSnapshot`

订单状态覆盖 created、submitted、accepted、partial、filled、cancelled、expired 和 rejected。`reduce_order_events` 使用稳定事件 ID 去重，并按时间确定性重放重复或乱序事件。

### 后端协议

`portfolio_backtester.backends` 公开：

- `BacktestBackend`
- `BackendCapabilities`
- `CanonicalBacktestResult`
- `BackendRegistry`
- `NativePositionReplayBackend`

每个后端都要声明订单生命周期、部分成交、每日账本和多空能力。缺少相应能力时，规范化结果中的相关表保持为空。

`NativePositionReplayBackend` 对当前持仓周期回放做安全收口：

- 拒绝 `long_only=False`、空头和负权重。
- 使用 `ffill` 成交价时要求显式允许过期价格。
- 使用分钟数据时要求声明信号在开盘前产生，或由调用方裁剪到可执行窗口。

## 内置后端保留能力

以下能力继续由内置实现维护：

- T+1 可卖数量。
- 买入整手和零股卖出。
- 涨跌停的方向限制。
- 停牌、上市、退市和最后估值。
- 带生效日期的费用表。
- 分红、拆股、送转等企业行动。
- PIT 输入和决策、下单、成交、估值时间戳。
- 输入、配置、日历、费率和校准版本哈希。

## 固定对照场景

后端变更至少覆盖三组固定场景：

1. 充分流动性多头场景，对比持仓、换手、毛收益、费用和净值。
2. A 股规则场景，覆盖 T+1、涨跌停、停牌、整手和最低佣金。
3. 容量与部分成交场景，覆盖多日成交、撤单重报、现金不足和非线性最低佣金。

未来的 Backtrader 适配器只有满足以下条件，才有资格进入 registry：

- 作为可选依赖安装。
- 转换为 `CanonicalBacktestResult`。
- 不向公开结果泄露第三方对象。
- 固定对照场景全部通过。
- A 股领域语义保持完整。
- 迁移和回滚路径经过评审。

达到这些条件代表适配器可以接受评审，不会自动替换内置实现。

## 统一账本路线

长期账本链路为：

```text
目标持仓 -> 订单 -> 成交 -> 持股与现金 -> 每日净值 -> 报告
```

当前 `NativePositionReplayBackend` 仍是周期级回放。它的 `orders`、`fills` 和 `daily_ledger` 为空，能力声明会如实反映这一限制。共享每日账本属于后续路线图，详见 [会计与执行路线图](../accounting_execution_roadmap.md)。

## 范围约束

本仓库不新增以下通用基础设施：

- 通用事件引擎。
- 通用 Gateway。
- 通用订单管理系统。
- 通用 CTA 策略基类。
- 通用参数优化器。
- 通用数据源。
- 与市场无关的券商模拟器。

项目代码继续聚焦组合领域、A 股市场语义、稳定契约和可信研究结果。
