# Optimizer Backend Boundary Design

## Goal

Create a stable portfolio-optimization boundary owned by `portfolio-backtester` so external solvers can be evaluated without leaking framework objects or replacing existing A-share execution semantics.

## Design

`PortfolioOptimizationRequest` carries pandas/native inputs: return history, optional expected returns, optional previous/benchmark weights, long-only bounds, and covariance shrinkage. `PortfolioOptimizationResult` returns only normalized weights and JSON-compatible diagnostics.

`OptimizerRegistry` provides explicit registration with no plugin discovery. Initial native baselines are equal weight and the repository's existing HRP implementation. Third-party adapters remain future optional dependencies and must emit the canonical result.

## Constraints

- No new runtime dependency in this PR.
- Native baselines remain deterministic and framework-neutral.
- No alpha-research or strategy-pipeline runtime import.
- Long-short optimization is outside this first boundary.
- A-share order execution, T+1, fees, market rules, and capacity remain in existing replay/execution layers.
- Future external adapters require fixed-scenario differential evidence and rollback instructions.
