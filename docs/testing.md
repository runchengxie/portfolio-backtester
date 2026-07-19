# 测试和质量检查

本页说明 `portfolio-backtester` 的本地测试入口、保留的远程检查模板和实际检查范围。

## 安装开发依赖

```bash
uv sync --locked --extra dev
```

项目使用 Python 3.12，依赖版本由 `uv.lock` 固定。

## 统一入口

```bash
scripts/dev/run_tests.sh <mode> [args...]
```

| 模式 | 实际范围 |
| --- | --- |
| `all` | 完整 `pytest` 测试集 |
| `fast` | `all` 的兼容别名 |
| `unit` | `all` 的兼容别名 |
| `lint` | Ruff 代码检查 |
| `format` | Ruff 格式检查 |
| `format-all` | `format` 的兼容别名 |
| `typecheck` | `ty` 配置范围 |
| `typecheck-release` | `typecheck` 的兼容别名 |
| `maintainability` | 维护性指标和当前预算 |

`fast` 和 `unit` 没有缩小测试范围。

## 常用命令

```bash
scripts/dev/run_tests.sh all
scripts/dev/run_tests.sh all tests/test_execution_contracts.py
scripts/dev/run_tests.sh all tests/test_backtest_backends.py
scripts/dev/run_tests.sh all -k position_backtest
scripts/dev/run_tests.sh lint
scripts/dev/run_tests.sh format
scripts/dev/run_tests.sh typecheck
scripts/dev/run_tests.sh typecheck-release
scripts/dev/run_tests.sh maintainability
```

## 推送前检查

在 `research-workspace` 受管检出中，顶层共享 `pre-push` 会按照工作区清单运行本仓库的导入检查、Ruff、格式检查、`ty` 和完整测试集。

单独克隆本仓库时不会继承共享钩子。推送前应手动运行上方列出的 `lint`、`format`、`typecheck`、`all` 和 `maintainability`。

## GitHub Actions 状态

`.github/workflows/ci.yml` 文件存在，并声明 pull request 与手动触发入口。仓库级 GitHub Actions 当前关闭，GitHub 不会为该文件创建运行记录。这个文件只作为轻量检查模板保留。本地命令和工作区 `pre-push` 是当前质量事实来源。

模板覆盖：

- Ruff 代码和格式检查。
- 当前登记范围的 `ty`。
- 框架中立的执行契约。
- 后端协议、规范化结果和 `native` 固定对照样例。
- 框架状态账本。
- 持仓回放回归。
- 包导入检查。

模板使用并发控制取消同一 pull request 的旧运行。重新启用仓库级 Actions 前，需要复核触发条件、配额和检查范围。完整测试和维护性预算继续由本地或工作区门禁执行。

## 类型检查范围

`ty` 只检查 `pyproject.toml` 中登记的文件。该范围已经合并迁移前的发布类型检查路径。检查通过只说明这些路径没有发现阻塞问题。

扩大类型覆盖时，应先修复目标模块，再更新配置和测试说明。新后端边界先由运行时契约测试保护，后续在共享账本迁移时纳入完整静态检查。

## 测试重点

当前测试集主要覆盖：

- Top-K 组合构造和收益计算
- `BacktestSpec` 序列化和历史入口一致性
- 持仓回放和退出规则
- 成本、滑点和交易约束
- 执行容量模拟
- 持仓契约和策略配置
- A 股整手约束
- benchmark、容量、暴露和报告
- 流动性代理、缓冲区和换手限制
- 框架中立的订单状态、重复事件和乱序事件归约
- 规范化后端结果与固定对照场景
- 包导入和跨仓库依赖隔离
- 维护性指标脚本

新增公开入口或修改输出契约时，应增加行为测试、固定对照样例和导入测试。
