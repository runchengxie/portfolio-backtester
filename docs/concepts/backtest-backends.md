# 回测后端和差分证据

本包通过 `BacktestBackend` 协议统一回测结果，不统一第三方框架的运行时对象。原生 A 股持仓回放、Qlib 和外部 LEAN 参考运行都在边界内转换成两张表：

- `performance` 按日期记录毛收益、净收益、换手、费用、滑点、总成本和累计 PnL
- `positions` 按日期和证券记录权重

跨仓库产物只保存这两张框架无关的表和 JSON 元数据。Qlib 的 Strategy、Executor、Position 和 LEAN 的对象不会进入公开结果契约。

## 原生 A 股回放

`NativeAShareReplayBackend` 包装已有的 `run_position_backtest`。它继续是 A 股语义的标准实现，负责不可交易状态、延迟退出、价格回退和费用口径。

```python
from cstree.backtesting import NativeAShareReplayBackend, PositionReplayRequest

request = PositionReplayRequest(
    positions=positions,
    pricing=pricing,
    periods=periods,
    config=config,
)
result = NativeAShareReplayBackend().run(request)
```

累计 `pnl` 从每期 `net_return` 复利计算。原生结果的 positions 表示调仓目标，日期使用输入中的调仓日期。收益日期使用实际退出日期。metadata 会把该口径记录为 `target_rebalance`。

## Qlib 后端

Qlib 是可选依赖：

```bash
uv sync --locked --extra dev --extra qlib
```

该 extra 将 pandas 限制在 2.x，并要求 MLflow 2.x 或 3.x。这两个约束只在安装 Qlib 时生效，用于避免 pyqlib 的宽松依赖范围解析到未验证组合。

适配器会延迟导入并直接调用官方 `qlib.backtest.backtest`。调用方仍负责初始化 Qlib、准备数据 Provider，以及提供可序列化的 Strategy 和 Executor 配置。

```python
from cstree.backtesting.integrations.qlib import (
    QlibBacktestBackend,
    QlibBacktestRequest,
)

request = QlibBacktestRequest(
    start_time='2026-01-01',
    end_time='2026-03-31',
    strategy={
        'class': 'TopkDropoutStrategy',
        'module_path': 'qlib.contrib.strategy.signal_strategy',
        'kwargs': {'topk': 20, 'n_drop': 4},
    },
    executor={
        'class': 'SimulatorExecutor',
        'module_path': 'qlib.backtest.executor',
        'kwargs': {'time_per_step': 'day', 'generate_portfolio_metrics': True},
    },
    frequency='1day',
)
result = QlibBacktestBackend().run(request)
```

Qlib 报告中的 `return` 表示扣费前收益，`cost` 表示成本。适配器将两者转换成 `gross_return`、`fee_cost` 和 `net_return`。Qlib 无法区分费用和滑点时，全部 `cost` 暂时归入 `fee_cost`，并在 metadata 中记录该分类规则。Qlib positions 来自账户的交易后快照，metadata 将口径记录为 `post_trade_account`。

适配器不会把原生 `PositionReplayRequest` 自动翻译成 Qlib Strategy。两者的调仓和执行时点需要由调用方显式配置，避免一个看似方便的转换层掩盖语义差异。

未安装 Qlib 时，导入 `cstree.backtesting` 和适配器模块都不会失败。只有实际运行 `QlibBacktestBackend` 时才会提示安装可选依赖。

## 差分报告

`compare_backtest_results` 比较五类信息：

1. 收益日期和持仓日期
2. 持仓证券及权重
3. 换手
4. 费用、滑点和总成本
5. 毛收益、净收益和累计 PnL

```python
from cstree.backtesting import (
    DifferenceDimension,
    DifferenceExplanation,
    compare_backtest_results,
)

report = compare_backtest_results(
    native_result,
    qlib_result,
    explanations={
        DifferenceDimension.DATES: DifferenceExplanation(
            code='execution_timestamp',
            detail='Qlib records the post-trade position date; native records the signal rebalance date.',
        ),
    },
)
```

没有差异的维度状态为 `matched`。存在差异且提供原因时状态为 `explained`。存在差异但没有原因时状态为 `unexplained`，此时 `report.accepted` 为 `False`。容差只处理浮点误差，不用于隐藏市场语义差异。

## LEAN 黄金基准交换

LEAN 不进入本包运行时。`cstree.backtesting.integrations.lean` 提供两个 JSON schema：

- `lean_golden_scenario.v1` 保存持仓、行情、持有期、回放配置和元数据
- `lean_golden_result.v1` 保存统一 performance、positions 和 fills

每个文件都使用稳定排序和 canonical JSON 计算 SHA-256。读取时会校验 schema 和内容 hash。数组中的表记录顺序不会影响 hash。

```python
from cstree.backtesting.integrations.lean import (
    LeanGoldenScenario,
    export_lean_scenario,
)

scenario = LeanGoldenScenario.from_position_replay_request(
    scenario_id='a-share-delayed-exit',
    description='不可交易证券延迟退出。',
    request=request,
)
scenario_hash = export_lean_scenario(scenario, 'scenario.json')
```

外部 LEAN 运行器读取 scenario，运行参考实现，再写回 result。Python 侧导入 result 后使用同一差分报告比较。仓库中的 `tests/fixtures/lean/` 提供了不可交易延迟退出和费用语义的原生黄金基准文件。

## 适用边界

差分相等不证明两个框架在全部策略上等价。每个黄金场景应覆盖一个清楚的市场语义，例如停牌、涨跌停、延迟退出、费用方向或调仓时点。新增差异说明时需要使用稳定的 code，并在 detail 中记录可复核的原因。
