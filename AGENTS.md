# AGENTS.md

本文件给维护者、外部贡献者和代码代理使用。它描述 `portfolio-backtester` 的本仓协作规则；工作区层面的规则仍以顶层 `research-workspace/AGENTS.md` 为准。

## 仓库范围

本仓库负责组合回测与持仓管理模块（`cstree.backtesting.*`），维护 Top-K 组合构造、调仓逻辑、执行模拟、容量和暴露报告、持仓后处理、换手归因、benchmark ladder 和回测报告。

本仓库可以消费外部信号、行情和 tradability 数据，但不应在运行时导入 alpha 研究（`cstree.alpha`）、策略编排（`cstree.pipeline`）或交易执行实现。研究编排和 `targets.json` 导出仍由 `strategy-pipeline` 负责。

## 常用命令

日常阻塞检查：

```bash
uv run --extra dev ruff check .
uv run --extra dev ruff format --check .
uv run --extra dev ty check
uv run --extra dev pytest
```

统一脚本入口：

```bash
scripts/dev/run_tests.sh lint
scripts/dev/run_tests.sh format
scripts/dev/run_tests.sh typecheck
scripts/dev/run_tests.sh all
```

发布前或诊断类型债时运行 BasedPyright：

```bash
scripts/dev/run_tests.sh basedpyright
```

GitHub Actions 中 Ruff、format、`ty check` 和维护性 ratchet 是阻塞检查，BasedPyright 是非阻塞建议项。

## GitHub 发布偏好

- 用户明确要求 commit、push 或发布本仓改动时，默认直接在 `main` 上提交并推送到 `origin/main`。
- 不要默认新建 `codex/*` 分支或 draft PR；只有用户明确要求 PR、远端规则阻止直接推送、工作区存在难以拆分的混杂改动，或改动风险需要人工 review 时才走分支和 PR。
- 本仓作为 `research-workspace` 子模块使用时，推送本仓后还要回到顶层仓库提交更新后的 submodule gitlink。

## 文档归属

新增组合或回测说明时，优先放在本仓 `docs/`：

- Top-K、buffer、分组约束、手数约束和持仓后处理。
- 回测收益、交易成本、换手、容量、暴露、benchmark ladder 和报告字段。
- `positions_by_rebalance.csv`、`positions_current*.csv` 及其下游消费约定。
- A 股 round-lot 可执行性、执行模拟和组合层敏感性分析。

留在 `strategy-pipeline` 的说明应聚焦编排、CLI、配置合成、运行目录和执行目标导出。

## 编辑规则

- 保持本仓可独立安装和测试，不通过 sibling source path 补齐 import。
- 不提交 `.pytest_cache/`、`__pycache__/`、`artifacts/`、`outputs/`、provider 凭证或本地 `.env*`。
- 修改持仓或回测产物契约时，同步更新 README、docs 和对应测试。
- 涉及跨仓库文件约定时，同步检查顶层 `research-workspace` 的 contract 文档和 submodule gitlink。
