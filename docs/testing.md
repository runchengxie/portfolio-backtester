# 测试和质量检查

本页说明 `portfolio-backtester` 的本地测试入口和实际检查范围。

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
scripts/dev/run_tests.sh all tests/test_execution.py
scripts/dev/run_tests.sh all -k position_backtest
scripts/dev/run_tests.sh lint
scripts/dev/run_tests.sh format
scripts/dev/run_tests.sh typecheck
scripts/dev/run_tests.sh maintainability
scripts/dev/run_tests.sh basedpyright
```

## 类型检查范围

`ty` 和 BasedPyright 只检查 `pyproject.toml` 中登记的文件。检查通过只说明这些路径没有发现阻塞问题。

扩大类型覆盖时，应先修复目标模块，再更新配置和测试说明。

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
- 包导入和跨仓库依赖隔离
- 维护性指标脚本

新增公开入口或修改输出契约时，应增加行为测试和导入测试。

## 自动化状态

当前仓库没有启用 GitHub Actions 测试 workflow。本地脚本和 `pyproject.toml` 是检查范围的事实来源。
