# portfolio-backtester

`portfolio-backtester` 是通用组合构造与回测工具包，权威 Python 包是 `portfolio_backtester`。

它接收外部信号、目标持仓和行情数据，提供：

- 组合构造和持仓回放
- 收益、成本和换手分析
- 新仓准入与旧仓退出分离的低换手组合构造
- 滑点、交易约束和执行容量模拟
- 基准、暴露和报告
- PSR、DSR 和多重试验 Sharpe 推断
- 输入输出契约与稳定高层 API
- 框架中立的执行契约、后端协议和规范化回测结果

仓库可以独立安装和测试，运行时不依赖 `strategy-pipeline` 或私有研究仓库。

## 安装

```bash
git clone https://github.com/runchengxie/portfolio-backtester.git
cd portfolio-backtester
uv sync --locked --extra dev
uv run --extra dev python -m pytest
```

项目使用 Python 3.12 或更高版本。

## 使用入口

常用高层入口包括：

```python
from portfolio_backtester import (
    PositionBacktestConfig,
    evaluate_position_backtest,
    run_position_backtest,
)
```

`run_position_backtest` 只汇总策略自身结果。需要信息比率、跟踪误差、alpha 和 beta 时，使用 `evaluate_position_backtest` 补充基准评估。

新仓准入和旧仓退出需要使用不同资格时，可以从包根调用旧仓再资格组合构造：

```python
from portfolio_backtester import (
    IncumbentRequalificationPolicy,
    select_incumbent_requalified_portfolio,
)

result = select_incumbent_requalified_portfolio(
    candidates,
    previous_symbols=previous_symbols,
    policy=IncumbentRequalificationPolicy(
        portfolio_size=20,
        entry_rank_limit=20,
        exit_rank_limit=40,
        max_new_positions=4,
        industry_cap=4,
    ),
)
```

新仓必须满足严格准入条件。旧仓会使用当日信息重新评分，并在更宽的退出缓冲区内继续持有。每日新增预算无法填满组合时，空缺权重保留为现金，不会重新分配给剩余持仓。完整语义见[旧仓再资格组合构造](docs/guides/incumbent-requalification.md)。

DailyWatch20 的组合策略、错位持有执行与稳定汇总也可以从包根导入：

```python
from portfolio_backtester import (
    EXECUTION_SUMMARY_SCHEMA,
    PORTFOLIO_POLICY_SCHEMA,
    DailyWatch20PortfolioPolicy,
    StaggeredCohortExecutionConfig,
    StaggeredCohortExecutionResult,
    execution_summary_frame,
    simulate_staggered_cohort_execution,
    summarize_staggered_execution,
)
```

错位持有执行按 `horizon_days` 建立同样数量的独立 cohort，并为每个 cohort 分配
`1 / horizon_days` 的初始资金。H1 只有一个 cohort，因此其收益就是整个 ledger 的收益。

输入表、最小示例和返回值说明见 [docs/guides/entry-points.md](docs/guides/entry-points.md)。

公开 API 与契约见：

- [公开 API](docs/reference/public-api.md)
- [旧仓再资格组合构造](docs/guides/incumbent-requalification.md)
- [组合式回测规范](docs/concepts/backtest-spec.md)
- [回测后端与统一账本边界](docs/concepts/backend-architecture.md)
- [持仓输出约定](docs/reference/outputs/positions.md)

## 回测后端

安全入口位于 `portfolio_backtester.backends`。它提供框架中立的后端协议、能力声明和规范化结果。当前主分支只有 `NativePositionReplayBackend` 实现，可以用它执行现有持仓周期回放：

```python
from portfolio_backtester.backends import (
    NativePositionReplayBackend,
    NativePositionReplayRequest,
)

result = NativePositionReplayBackend().run(
    NativePositionReplayRequest(
        positions=positions,
        pricing=pricing,
        periods=periods,
        config=config,
    )
)
```

该入口会拒绝空头、负权重和 `long_only=False`。分钟 VWAP 与过期的 `ffill` 成交价需要显式声明研究假设。

当前 registry 只包含 `native.position_replay`。Qlib 与 LEAN 的历史候选没有进入 `main`，其中 LEAN 只保留架构参考价值。Backtrader 仍处于规划阶段。vn.py 属于本仓库范围外。状态和退役门禁见 [后端架构](docs/concepts/backend-architecture.md)与[机器可读集成账本](docs/framework-integration-ledger.yml)。

## 开发检查

```bash
scripts/dev/run_tests.sh lint
scripts/dev/run_tests.sh format
scripts/dev/run_tests.sh typecheck
scripts/dev/run_tests.sh typecheck-release
scripts/dev/run_tests.sh all
scripts/dev/run_tests.sh maintainability
```

`fast` 和 `unit` 是 `all` 的兼容别名，都会运行完整测试集。`typecheck-release` 与 `typecheck` 都运行 `ty`，检查范围相同。

仓库保留 `.github/workflows/ci.yml` 作为轻量检查模板，仓库级 GitHub Actions 当前关闭，因此不会产生远程运行记录。本地命令和工作区 `pre-push` 是当前质量事实来源。

详细范围见 [docs/testing.md](docs/testing.md)。

## 仓库边界

本仓库维护通用组合构造、回测、成本、容量、暴露、报告、回测统计推断，以及框架中立的结果和执行契约。

数据采集、特征工程、模型训练、具体策略规则、任务编排和券商执行由调用方或其他仓库负责。Gateway 和实时传输不在本仓库职责内。第三方框架对象不得进入本仓库公开结果或跨仓库产物。

`DailyWatch20` 是现有调用方使用的兼容例外。本仓库只保留其组合选择、组合策略、错位持有执行与回执接口，研究假设、特征和晋升证据由 `alpha-research`、`research-apps` 与 `strategy-pipeline` 维护。新增策略专用规则不应继续扩展这一例外。

工作区 2.0 已删除旧共享 namespace 和 facade。新代码只使用 `portfolio_backtester`。

## Python 包名

权威包名为 `portfolio_backtester`。工作区 2.0 已移除 1.x 兼容包名和旧门面接口，导入、契约、产物类型、日志名称和环境变量均使用本仓库命名。

## 文档入口

- [文档首页](docs/README.md)
- [常用入口](docs/guides/entry-points.md)
- [旧仓再资格组合构造](docs/guides/incumbent-requalification.md)
- [回测后端与统一账本边界](docs/concepts/backend-architecture.md)
- [成本与执行假设](docs/concepts/execution-costs.md)
- [执行容量与每日净值模拟](docs/guides/execution-simulation.md)
- [AFML 仓位与策略风险](docs/concepts/afml-sizing-and-risk.md)
- [换手率口径](docs/concepts/turnover.md)
- [成本口径](docs/concepts/cost-breakdown.md)
- [测试和质量检查](docs/testing.md)

请勿提交凭证、账户信息、未授权数据、`artifacts/` 或 `outputs/`。

仓库当前没有许可证文件。公开可见不自动授予复制、修改或再分发权限。
