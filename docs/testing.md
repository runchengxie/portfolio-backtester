# 测试和质量检查

本页说明 `portfolio-backtester` 的本地测试入口、远程 PR 检查和实际检查范围。

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
| `basedpyright` | BasedPyright 配置范围 |
| `typecheck-release` | `basedpyright` 的兼容别名 |
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
scripts/dev/run_tests.sh maintainability
scripts/dev/run_tests.sh basedpyright
```

## 推送前检查

在 `research-workspace` 受管检出中，顶层共享 `pre-push` 会按照工作区清单运行本仓库的导入检查、Ruff、格式检查、`ty` 和完整测试集。

单独克隆本仓库时不会继承共享钩子。推送前应手动运行上方列出的 `lint`、`format`、`typecheck`、`all`、`maintainability` 和 `basedpyright`。

## 远程 PR CI

启用仓库级 GitHub Actions 后，`.github/workflows/ci.yml` 会在 pull request 和手动触发时运行。它是轻量质量门禁，不替代完整本地检查。仓库级 Actions 关闭时，工作流文件会保留，但 GitHub 不会创建运行记录。

远程 CI 覆盖：

- Ruff lint 和格式检查；
- 当前登记范围的 `ty`；
- framework-neutral execution contracts；
- backend protocol、canonical result 和 native golden fixture；
- framework integration ledger；
- position backtest 回归；
- package import smoke。

工作流使用 concurrency 取消同一 PR 的旧运行，避免把 Actions 配额献给已经过时的提交。完整测试、BasedPyright 和维护性预算仍由本地或工作区门禁执行。

## 类型检查范围

`ty` 和 BasedPyright 只检查 `pyproject.toml` 中登记的文件。检查通过只说明这些路径没有发现阻塞问题。

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
- framework-neutral 订单状态、重复和乱序事件归约
- canonical backend 结果与 golden scenario
- 包导入和跨仓库依赖隔离
- 维护性指标脚本

新增公开入口或修改输出契约时，应增加行为测试、golden fixture 和导入测试。
