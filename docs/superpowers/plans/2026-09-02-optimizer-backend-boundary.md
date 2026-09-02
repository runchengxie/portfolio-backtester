# Optimizer Backend Boundary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an owner-controlled optimizer request/result boundary with native equal-weight and HRP baselines.

**Architecture:** Keep optimization separate from backtest/execution backends. External solver adapters translate to/from pandas/native platform types and never escape third-party objects through public APIs.

**Tech Stack:** Python 3.12, pandas, NumPy, existing HRP implementation, pytest.

**Spec:** `docs/superpowers/specs/2026-09-02-optimizer-backend-boundary-design.md`

## Global Constraints

- No new runtime dependency.
- Existing HRP implementation is reused, not copied.
- New request/result types remain framework-neutral.
- Long-only is the only supported native optimization mode in this PR.

---

### Task 1: Define tests and canonical types

**Files:**
- Create: `tests/test_optimizer_backends.py`
- Create: `src/portfolio_backtester/optimization.py`

- [x] Write tests for equal weight, HRP bounds, registry behavior, asset mismatch, and infeasible bounds.
- [ ] Run `uv run --extra dev pytest tests/test_optimizer_backends.py -q` and confirm RED before implementation.
- [x] Implement request/result validation, registry, equal-weight baseline, and HRP adapter.
- [ ] Run focused tests and confirm PASS.

### Task 2: Publish the owner API

**Files:**
- Modify: `src/portfolio_backtester/__init__.py`
- Modify: `tests/test_package_smoke.py`
- Create: `docs/concepts/portfolio-optimization-backends.md`

- [x] Export the optimizer types and native backends.
- [x] Register the new module and exports in package smoke tests.
- [x] Document external-adapter adoption order.
- [ ] Run `uv run --extra dev pytest tests/test_optimizer_backends.py tests/test_package_smoke.py -q`.

### Task 3: Repository gates

- [ ] Run `scripts/dev/run_tests.sh lint`.
- [ ] Run `scripts/dev/run_tests.sh format`.
- [ ] Run `scripts/dev/run_tests.sh typecheck`.
- [ ] Run `scripts/dev/run_tests.sh all`.
- [ ] Run `scripts/dev/run_tests.sh maintainability` and separate any pre-existing ratchet failure from this change.
