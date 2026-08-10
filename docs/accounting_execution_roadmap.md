# 会计与执行路线图

本页记录信号回测、持仓回放和容量分析逐步共用一套可审计账本的计划。当前已完成框架中立的执行契约、后端结果边界、术语与结果契约，订单级共享账本引擎的初步接入，以及第三阶段的成本子项拆分（佣金/印花税/过户费/价差真实拆分，冲击/机会/融资为占位 0）。第四阶段及之后的市场规则、容量校准与复现元数据仍属于路线图，不能当作现有能力使用。

## 长期约束

各类回测最终应遵守同一组约束：

1. 目标权重生成带方向的订单。
2. 订单生成零笔或多笔成交。
3. 成交更新持股数量和现金。
4. 每日净值等于现金与持仓市值之和。
5. 显式费用和隐含执行成本分别报告。
6. 所有入口使用有文档说明的统一换手率公式。
7. 已配置的执行输入缺失时终止计算，不能静默关闭约束。
8. 第三方框架对象不能进入公开结果或跨仓库产物。
9. 后端必须声明订单、部分成交、每日账本和多空能力，不能伪造不具备的输出。

## 第零阶段：执行契约与后端边界

当前已经完成：

- 定义 `Instrument`、`Target`、`OrderIntent`、`OrderEvent`、`Fill` 和 `LedgerSnapshot`。
- 定义 created、submitted、accepted、partial、filled、cancelled、expired 和 rejected 状态。
- 使用稳定事件 ID 去重，并对乱序订单事件做确定性归约。
- 定义 `BacktestBackend`、`BackendCapabilities` 和 `CanonicalBacktestResult`。
- 提供 `NativePositionReplayBackend` 作为现有周期回放的安全适配器。
- 用机器可读账本分别记录当前内置后端、历史候选、规划项和范围外项目。
- 增加充分流动性的 `native` 固定对照样例和本地质量门禁。

该阶段只建立稳定边界，没有把周期回放升级为订单级或每日账本。`orders`、`fills` 和 `daily_ledger` 在 native 周期回放中保持为空，并通过 capabilities 明确声明不可用。

## 第一阶段：术语与结果契约

当前已经完成：

- 区分持仓名称替换率和权重换手率
- 输出买入、卖出、总交易权重、半 L1 换手率和单边换手率
- 保留历史建仓成本约定，同时单独披露半 L1 换手率
- 在单腿和分期结果中提供有类型约束的成本明细
- 在用户文档中说明会计公式

## 第二阶段：共享每日账本

状态：已完成（可选 ledger 模式，默认关闭以保旧契约）。已落地独立的订单级账本引擎 `execution_sim/`，`core.py` 真实产出 `orders` 与 `fills`，并已接入 `run_position_backtest`（`_evaluation_positions.py` 把 `sim_result.orders/fills/daily` 写入 `execution_sim_*` 字段，`simulate_ideal_daily_nav` 也已实现）。同时新增了 `to_unified_ledger()` 适配方法，把执行引擎的 `daily/orders/fills` 规范映射为路线图定义的八个统一账本字段：`targets`、`orders`、`fills`、`daily_positions`、`daily_cash`、`daily_nav`、`cost_breakdown`、`turnover_breakdown`。两项原本规划、尚未完成的内容现已通过可选 ledger 开关接入，具体如下。

- 旧 `native` 周期回放后端保留历史契约（默认 `orders/fills/daily_ledger` 仍声明为 `not_available`）。新增可选 `ledger` 开关（`NativePositionReplayRequest.ledger`），开启时调用 `execution_sim` 的 `simulate_execution_adjusted_nav`，把 `orders/fills/daily_ledger` 填入 `CanonicalBacktestResult` 并翻转对应 `capabilities`（含 `order_id`、`fill_id`，满足框架中立账本契约）。默认关闭，固定对照测试不受影响。
- `backtest_topk`（api.py）保留原有的 `ExecutionModel` facade 与五元素返回契约。新增可选 `ledger` 开关，开启时复用引擎累积的每期目标权重（`target_weights`），通过 `simulate_ideal_daily_nav` 产出统一账本，并在返回包末尾追加 `UnifiedLedger` 元素。默认关闭，旧签名与固定场景差分不受影响。

原规划让 `backtest_topk`、`run_position_backtest`、理想净值和容量调整净值共用以下账本链路：

```text
目标持仓 -> 订单 -> 成交 -> 持股与现金 -> 每日净值 -> 报告
```

信号回测届时只负责构造目标持仓，会计计算交给与外部持仓回放相同的引擎。计划统一输出：

- `targets`
- `orders`
- `fills`
- `daily_positions`
- `daily_cash`
- `daily_nav`
- `cost_breakdown`
- `turnover_breakdown`

实施顺序：

1. 先迁移 `simulate_ideal_daily_nav`，建立现金和持仓守恒基准。
2. 再迁移 `run_position_backtest`，保留旧周期结果作为兼容视图。
3. 将 score-driven 回测限制为目标持仓生成。
4. 最后迁移容量模拟的订单、部分成交和 cancel/replace。

## 第三阶段：成本拆分

已完成（执行路径已真实拆分，8 个互不重复子项全部接入。无模型的子项用 0 占位）。

`CostBreakdown`（`src/portfolio_backtester/types.py`）包含 8 个互不重复的子项：佣金、印花税、过户费、价差成本、临时冲击、永久冲击、机会成本、融资成本。聚合关系保持为 `fee_cost = 佣金 + 印花税 + 过户费`，`slippage_cost = 价差成本 + 临时冲击 + 永久冲击 + 机会成本 + 融资成本`，`total_cost = fee_cost + slippage_cost`（即 8 个子项之和）。原有 `fee_cost`/`slippage_cost`/`total_cost` 语义与 `to_dict` 全部保留。`from_components` 在有分项数据时构造，`to_unified_ledger()` 的 `cost_breakdown` 现在输出各子项，并额外给出 `fee_cost`/`slippage_cost`/`transaction_cost` 聚合列。

### 执行路径真实接入（本阶段落地）

- `DetailedTradeFeeModel`（`src/portfolio_backtester/_execution_models.py`）新增 `notional_cost_breakdown(notional, side)`，按 A 股口径返回 `commission`/`stamp_tax`/`transfer_fee`/`spread_cost` 四项，且四项之和恒等于 `notional_cost`（守恒）。
- 执行引擎的每笔成交成本由单值 `_trade_fee` 改为返回 `CostBreakdown`：`orders_ideal.py`、`orders_nav.py` 的 ideal/adjusted-nav 路径均按子项累加，并通过 `_record_nav_fill` 把子项写入 `fills` 的 `cost_*` 列。每日 `daily` 行也带 `cost_*` 子项。
- `UnifiedLedger.to_unified_ledger()`（`execution_sim/results.py`）的 `cost_breakdown` 改为按买卖方向聚合并填入子项，再派生 `fee_cost`/`slippage_cost`/`transaction_cost`，保证 8 子项之和 == 旧 `fee_cost + slippage_cost` == 旧 `transaction_cost`。

### 子项现状

- 佣金、印花税、过户费、价差成本：已真实拆分（带 `DetailedTradeFeeModel` 时非零。无费率模型时按 `notional * cost_rate` 整体归入 `spread_cost`，保持总现金扣减不变）。
- 临时冲击、永久冲击、机会成本、融资成本：执行引擎尚无对应模型，统一以 0 占位（诚实占位，不编造数值）。

最低佣金需要明确计费单位。默认建议按股票、买卖方向和交易日分别计费，同时允许券商专用规则覆盖。

需要额外验证现金缩放与最低佣金的非线性组合，确保每次成交后现金不为负。

## 第四阶段：市场规则与时间戳

已完成（opt-in 契约，默认全部关闭以保持现有固定对照场景基线）。

市场规则契约通过 `ExecutionSimConfig` 新增开关接入，默认不约束，现有 7 个固定对照场景数值不变：

- 买入整手（`round_lot`，非 None 时买入数量向下取整到整手，不足一手当日不买）、零股卖出（卖出路径不受整手约束）。
- T+1 可卖数量（`enforce_t1`，当日可卖数量取自当日开盘前持仓快照 `t1_available`，排除当日新买）。
- 涨跌停与方向相关可交易状态（`enforce_price_limits` + `limit_up_col`/`limit_down_col` 布尔列，涨停跳过买入、跌停跳过卖出）。
- 上市、停牌、退市（`enforce_listing_status` + `listing_status_col`，非 `listed` 状态当日跳过成交）。
- 企业行动（分红、拆股、送转）：已预留输入契约与占位钩子，真实数值调整待数据源就绪后接入（与阶段三冲击/机会成本 0 占位风格一致，不编造数值）。

约束遵守严格按路线图长期约束 #7：任何市场规则若开启但所需输入列缺失，运行终止（`ValueError`）而非静默降级。同时运行 summary 增加 `warnings` 字段，当引擎启用但全部市场规则关闭时记录 `market_rules_inactive`，满足'不静默关闭约束'的可见性要求。

信号时间、决策时间、下单时间、成交时间和估值时间已分别记录在每笔 fill 行（`signal_time`/`decision_time`/`order_time`/`fill_time`/`valuation_time`），统一带 `Asia/Shanghai` 时区。前视偏差检查与成交价/估值价区分作为后续细化项保留。带生效日期的费用表（date-effective fee schedule）随企业行动契约一并预留。

## 第五阶段：容量与冲击校准

容量约束应直接限制成交量。超过参与率上限的部分需要形成未成交订单，或明确标记为外推结果。

容量报告计划补充：

- 盈亏平衡资金规模
- 成交率达到 95% 时的资金规模
- 达到指定 alpha 保留率时的资金规模
- 每增加一单位资金的边际冲击
- 按股票、行业和流动性分组统计的容量占用集中度

策略只在较窄时段交易时，应优先使用对应执行窗口的流动性，避免直接使用全日成交额。

## 第六阶段：指标与复现

收益和风险指标应统一从每日净值推导。自然周期汇总需要先在周期内复利，并保留年份维度。

每次运行计划保存：

- 仓库提交号
- 后端名称和 capability 快照
- 配置哈希
- 输入数据哈希
- 股票池和交易日历版本
- 费率表与滑点校准版本
- 依赖包版本
- 随机种子
- 运行时间

## 外部后端评审原则

当前没有外部后端适配器。Qlib 与 LEAN 的历史候选没有进入 `main`，LEAN 只保留架构参考用途。Backtrader 仍处于规划阶段。vn.py 属于本仓库范围外。

未来的 Backtrader 适配器需要保持可删除，并通过规范化结果转换进入现有后端协议。`native.position_replay` 在对照证据被接受前保持权威实现。

删除重复 native 通用实现前必须同时满足：

- 覆盖率达到 `framework-integration-ledger.yml` 的门槛。
- 固定对照场景全部通过。
- 日期、持仓、换手、成本、成交和 PnL 差异已分类。
- A 股领域语义无损。
- 性能、迁移说明和回滚路径达到约定。

## 验证要求

统一账本落地时至少需要以下测试：

- 现金与持仓市值守恒
- 零收益、零成本下的净值恒等关系
- 成本和容量的单调性
- Top-K 目标持仓与持仓回放结果等价
- 订单级最低佣金
- 人工核对过的每日现金、持股和净值样例
- 稀疏权重、价格缺失和延迟成交的性质测试
- 重复和乱序订单事件的幂等归约
- 可选适配器不泄露第三方对象
- `native` 与获准适配器的固定场景差分
