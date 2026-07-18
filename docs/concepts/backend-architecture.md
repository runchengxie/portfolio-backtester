# 回测后端与统一账本边界

本页定义 `portfolio-backtester` 在 native、Qlib、Backtrader 和 vn.py 之间的职责边界。目标是停止扩张通用交易框架能力，同时保留横截面组合研究、A 股市场语义、稳定产物和可审计结果契约。

## 核心决策

`portfolio-backtester` 是组合领域层、标准结果层和多后端调度边界，不绑定单一第三方框架。

```text
信号或外部目标持仓
        |
        v
组合选择、权重与约束
        |
        v
framework-neutral BacktestRequest
        |
        +-------------------+-------------------+
        |                   |                   |
        v                   v                   v
     native              Qlib              Backtrader
        |                   |                   |
        +-------------------+-------------------+
                            |
                            v
                CanonicalBacktestResult
                            |
                            v
                分析、差分、报告与产物
```

vn.py 不作为组合研究内核。它属于 `quant-execution-engine` 的可选 transport，用于 Gateway、paper/shadow 订单传输和回报事件归一化。

## 为什么不做纯 wrapper

Backtrader 和 vn.py 擅长订单生命周期、撮合、事件分发、策略回调、Gateway 和账户状态。当前项目的主要价值则集中在：

- 横截面 Top-K、缓冲区、分组上限和换手控制。
- 目标权重、容量、暴露和研究统计。
- A 股 T+1、方向相关可交易性、整手、费用和退市等语义。
- point-in-time 输入、文件契约、lineage 和可复现证据。

把所有横截面研究强行塞进事件策略框架，会增加目标权重与订单对象之间的双向转换，同时仍需保留大部分领域代码。后端可替换可以复用成熟框架。纯 wrapper 容易把框架对象变成新的内部依赖。

## 新增的稳定边界

### execution contracts

`portfolio_backtester.execution_contracts` 定义 framework-neutral 类型：

- `Instrument`
- `Target`
- `OrderIntent`
- `OrderEvent`
- `Fill`
- `LedgerSnapshot`

订单状态覆盖 created、submitted、accepted、partial、filled、cancelled、expired 和 rejected。`reduce_order_events` 使用 `event_id` 去重，并按时间稳定重放，以便测试重复和乱序回调。

### backend protocol

`portfolio_backtester.backends` 提供：

- `BacktestBackend`
- `BackendCapabilities`
- `CanonicalBacktestResult`
- `BackendRegistry`
- `NativePositionReplayBackend`

后端必须明确声明是否真正支持订单生命周期、部分成交、每日账本和多空。没有这些能力的后端必须输出空表，不能把模型估计冒充实际订单或成交。

`NativePositionReplayBackend` 是现有持仓周期回放的兼容适配器。它故意 fail closed：

- 拒绝 `long_only=False`、short side 和负权重。
- `ffill` 作为成交价时要求显式 opt-in。
- 使用分钟数据时要求声明信号在开盘前产生，或数据已经由调用方裁剪到可执行窗口。

旧高层 API 暂时保持兼容，调用方可以逐步迁移到安全后端入口。

## 各后端职责

| 后端 | 负责 | 不负责 |
| --- | --- | --- |
| native | 横截面目标、确定性持仓回放、A 股市场规则、成本和容量研究 | 通用 Gateway、实时 OMS、通用 CTA 框架 |
| Qlib | 通用 Top-K 基线、研究实验和差分回测 | A 股权威市场规则、跨仓库产物类型 |
| Backtrader | 事件驱动参考场景、订单类型和模拟时钟差分 | 所有横截面组合研究的强制运行时 |
| vn.py | Gateway、paper/shadow transport、订单成交账户事件 | `portfolio-backtester` 的组合构造和审计权威账本 |

机器可读状态和替换门禁见 `docs/framework-integration-ledger.yml`。

## 统一账本目标

长期目标仍是：

```text
目标持仓 -> 订单 -> 成交 -> 持股与现金 -> 每日净值 -> 报告
```

统一结果至少包含：

- performance
- positions
- orders
- fills
- daily ledger
- summary
- metadata

本 PR 先建立稳定类型和能力声明。后续按以下顺序迁移：

1. 将 `simulate_ideal_daily_nav` 迁移到共享账本。
2. 将 `run_position_backtest` 的会计计算迁移到同一账本。
3. 将 score-driven 回测限制为`信号到目标持仓`。
4. 将 capacity simulator 的订单和成交迁移到统一状态机。
5. 接入 Qlib 和 Backtrader 差分适配器。
6. 达到替换门禁后删除重复的 native 通用实现。

## 必须保留的 native 能力

以下能力在外部后端提供等价、可验证实现前保持 native canonical：

- T+1 可卖数量。
- 买入整手和零股卖出。
- 涨跌停方向限制。
- 停牌、上市、退市和最后估值。
- 带生效日期的费用表。
- 分红、拆股、送转等企业行动。
- PIT 和决策、下单、成交、估值时间戳。
- 输入、配置、日历、费率和校准版本哈希。

## golden scenarios

后端替换不能靠`示例跑通`判断。至少维护三组固定场景：

1. 充分流动性多头：比较持仓、换手、毛收益、费用和 NAV。
2. A 股规则：覆盖 T+1、涨跌停、停牌、整手和最低佣金。
3. 容量与部分成交：覆盖多日成交、cancel/replace、现金不足和非线性最低佣金。

外部后端只有同时满足以下条件，才可以替换对应 native 通用实现：

- 覆盖率达到机器可读账本中的门槛。
- golden scenarios 全部通过。
- 差异已分类且被接受。
- 性能达到文档阈值。
- A 股领域语义没有丢失。
- 迁移和回滚路径完整。

## 禁止继续扩张的方向

除非新的 ADR 明确推翻本决策，本仓库不再新增：

- 通用 EventEngine。
- 通用 Gateway。
- 通用 OMS。
- 通用 CTA Strategy 基类。
- 通用参数优化器。
- 通用 data feed。
- 与市场无关的 Broker 模拟器。

这些能力优先由成熟框架承担。项目代码应继续集中在组合领域、A 股语义、稳定契约和可信研究结果上。
