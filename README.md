# portfolio-backtester

`portfolio-backtester` 是通用组合构造与回测工具包，权威 Python 包是 `portfolio_backtester`。

它接收外部信号、目标持仓和行情数据，提供：

- 组合构造和持仓回放
- 收益、成本和换手分析
- 滑点、交易约束和执行容量模拟
- 基准、暴露和报告
- PSR、DSR 和多重试验 Sharpe 推断
- 输入输出契约与稳定高层 API
- framework-neutral 执行契约、可替换后端和差分回测边界

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

输入表、最小示例和返回值说明见 [docs/guides/entry-points.md](docs/guides/entry-points.md)。

公开 API 与契约见：

- [公开 API](docs/reference/public-api.md)
- [组合式回测规范](docs/concepts/backtest-spec.md)
- [回测后端与统一账本边界](docs/concepts/backend-architecture.md)
- [持仓输出约定](docs/reference/outputs/positions.md)

## 可替换后端

新的安全入口位于 `portfolio_backtester.backends`。它提供 framework-neutral 的后端协议、能力声明和 canonical result。现有持仓周期回放可以通过 `NativePositionReplayBackend` 进入该边界：

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

该入口会拒绝兼容 API 过去可能静默缩窄的 short side、负权重和 `long_only=False`。分钟 VWAP 与 stale `ffill` 成交价也必须显式声明研究假设。

后端职责、Backtrader/Qlib/vn.py 边界和 native 退役门禁见[后端架构](docs/concepts/backend-architecture.md)与[机器可读集成账本](docs/framework-integration-ledger.yml)。

## 开发检查

```bash
scripts/dev/run_tests.sh lint
scripts/dev/run_tests.sh format
scripts/dev/run_tests.sh typecheck
scripts/dev/run_tests.sh all
scripts/dev/run_tests.sh maintainability
```

`fast` 和 `unit` 是 `all` 的兼容别名，都会运行完整测试集。

BasedPyright 用于补充诊断：

```bash
scripts/dev/run_tests.sh basedpyright
```

启用仓库级 GitHub Actions 后，pull request 会运行轻量远程 CI，包括 Ruff、格式、现有类型检查、后端契约测试、持仓回放回归测试和包导入检查。完整测试集与维护性门禁仍在本地或工作区 pre-push 中运行。

详细范围见 [docs/testing.md](docs/testing.md)。

## 仓库边界

本仓库维护通用组合构造、回测、成本、容量、暴露、报告、回测统计推断，以及 framework-neutral 的结果和执行契约。

数据采集、特征工程、模型训练、具体策略规则、任务编排和券商执行由调用方或其他仓库负责。vn.py Gateway 和实时 transport 属于 `quant-execution-engine`。第三方框架对象不得进入本仓库公开结果或跨仓库产物。

`DailyWatch20` 是现有调用方使用的兼容例外。本仓库只保留其组合选择与回执接口，研究假设、特征和晋升证据由 `alpha-research` 与 `strategy-pipeline` 维护。新增策略专用规则不应继续扩展这一例外。

工作区 2.0 已删除旧共享 namespace 和 facade。新代码只使用 `portfolio_backtester`。

## Python 包名

权威包名为 `portfolio_backtester`。工作区 2.0 已移除 1.x 兼容包名和旧门面接口，导入、契约、产物类型、日志名称和环境变量均使用本仓库命名。

## 文档入口

- [文档首页](docs/README.md)
- [常用入口](docs/guides/entry-points.md)
- [回测后端与统一账本边界](docs/concepts/backend-architecture.md)
- [成本与执行假设](docs/concepts/execution-costs.md)
- [执行容量与每日净值模拟](docs/guides/execution-simulation.md)
- [AFML 仓位与策略风险](docs/concepts/afml-sizing-and-risk.md)
- [换手率口径](docs/concepts/turnover.md)
- [成本口径](docs/concepts/cost-breakdown.md)
- [测试和质量检查](docs/testing.md)

请勿提交凭证、账户信息、未授权数据、`artifacts/` 或 `outputs/`。

仓库当前没有许可证文件。公开可见不自动授予复制、修改或再分发权限。
