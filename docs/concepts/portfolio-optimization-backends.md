# Portfolio optimization backend boundary

`portfolio-backtester` now exposes a framework-neutral optimization request/result boundary before adopting any third-party solver.

## Why this exists

The repository already owns portfolio construction, HRP, turnover, execution costs, capacity, exposures, and A-share replay semantics. External optimizers should therefore plug into an owner-controlled boundary instead of exporting CVXPY, PyPortfolioOpt, cvxportfolio, Riskfolio, or Ricequant-specific objects into calling repositories.

The first two backends are deliberately boring baselines:

- `native.equal_weight`
- `native.hrp`

They provide fixed reference behavior and reuse the existing HRP implementation. No new solver dependency is introduced.

## Stable request

`PortfolioOptimizationRequest` carries only platform types:

- historical returns;
- optional expected returns;
- optional previous weights;
- optional benchmark weights;
- long-only weight bounds;
- covariance shrinkage.

The optional alpha/previous/benchmark inputs are included now so later constrained optimizers do not need to invent incompatible request types. Native baselines do not pretend to use inputs they do not need.

## Stable result

`PortfolioOptimizationResult` contains:

- backend name;
- normalized portfolio weights;
- JSON-compatible diagnostics;
- versioned result schema.

Validation enforces asset identity, finite weights, sum-to-one, long-only semantics, and request bounds.

## Adoption order

1. Keep equal weight / rank / sleeve / HRP as baselines.
2. Add fixed-scenario tests for the optimizer boundary.
3. Evaluate a PyPortfolioOpt adapter for conventional constrained optimization.
4. Evaluate cvxportfolio for cost-aware and multi-period research after license review.
5. Keep Riskfolio-Lib as a research comparison backend until its larger model surface has pre-registered use cases.
6. Treat RQOptimizer as a domain-design reference for benchmark, style/industry, tracking-error, turnover, and transaction-cost constraints; do not put proprietary objects in public contracts.

A new adapter must remain optional, emit `PortfolioOptimizationResult`, document its solver/version, have fixed-scenario differential evidence, and leave native baselines runnable when the dependency is absent.
