# 测试和质量检查

本页说明本地测试入口、各命令的实际范围和 GitHub Actions 的阻塞规则。

## 安装开发依赖

```bash
uv sync --locked --extra dev
```

仓库使用 Python 3.12。依赖版本由 `uv.lock` 固定。

## 统一入口

推荐通过下面的脚本运行检查：

```bash
scripts/dev/run_tests.sh <mode> [args...]
```

支持的模式如下：

| 模式 | 实际行为 |
| --- | --- |
| `all` | 运行完整的 `pytest` 测试集 |
| `fast` | `all` 的兼容别名，仍会运行完整测试集 |
| `unit` | `all` 的兼容别名，仍会运行完整测试集 |
| `lint` | 使用 Ruff 检查整个仓库 |
| `format` | 使用 Ruff 检查整个仓库的格式 |
| `format-all` | `format` 的兼容别名 |
| `typecheck` | 使用 `ty` 检查 `pyproject.toml` 配置的类型覆盖范围 |
| `basedpyright` | 使用 BasedPyright 检查配置的类型覆盖范围 |
| `typecheck-release` | `basedpyright` 的兼容别名 |
| `maintainability` | 检查文件长度、函数长度和复杂度等维护性指标 |

`fast` 和 `unit` 目前没有缩小测试范围。使用这些名称不会节省执行时间。脚本保留它们是为了兼容共享的持续集成调用方式。

## 常用命令

运行完整测试：

```bash
scripts/dev/run_tests.sh all
```

运行单个测试文件：

```bash
scripts/dev/run_tests.sh all tests/test_execution.py
```

按名称筛选测试：

```bash
scripts/dev/run_tests.sh all -k position_backtest
```

运行代码质量检查：

```bash
scripts/dev/run_tests.sh lint
scripts/dev/run_tests.sh format
scripts/dev/run_tests.sh typecheck
scripts/dev/run_tests.sh maintainability
```

查看 BasedPyright 诊断：

```bash
scripts/dev/run_tests.sh basedpyright
```

## GitHub Actions

`.github/workflows/tests.yml` 会运行下面的步骤：

1. 安装锁定的开发依赖
2. 运行 Ruff 代码检查
3. 运行 Ruff 格式检查
4. 运行 `ty` 类型检查
5. 运行完整的 `pytest` 测试集
6. 运行 BasedPyright
7. 运行维护性指标检查

BasedPyright 当前设置为建议项，失败时不会阻止工作流通过。其他步骤会阻止合并。

## 当前类型检查范围

`ty` 当前只检查 `src/cstree/backtesting/types.py`。

BasedPyright 检查 `pyproject.toml` 中列出的若干核心文件。它还没有覆盖整个 `src/cstree/backtesting` 目录。

因此，类型检查通过表示已配置的范围没有发现阻塞问题，不能推断整个包已经完成静态类型验证。

## 测试覆盖重点

当前测试集覆盖的主要领域包括：

- Top-K 回测和收益计算
- `BacktestSpec` 序列化、黄金结果和历史入口一致性
- 核心安装元数据不依赖 scikit-learn 或 XGBoost
- 核心导入、产品延迟加载和历史模块兼容
- 持仓回放和退出规则
- 成本模型、滑点模型和交易约束
- 执行容量模拟
- 持仓契约和策略配置
- A 股整手约束和诊断
- benchmark、容量、暴露和报告
- 流动性代理、缓冲区和换手限制
- 包导入和跨仓库依赖隔离
- 维护性指标脚本

新增公开入口或修改输出契约时，应同时增加行为测试和导入测试。
