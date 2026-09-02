# Factor Attribution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Add active return/risk decomposition that later consumes the platform risk model and publishes attribution evidence.

**Spec:** `docs/superpowers/specs/2026-09-02-factor-attribution-design.md`

- [x] Add tests for return reconciliation, risk reconciliation, and asset/factor mismatch.
- [ ] Run focused tests and confirm RED before implementation.
- [x] Implement factor return contribution, specific residual, cost drag, factor variance contribution, and specific variance contribution.
- [x] Document why Brinson attribution remains a separate semantic layer.
- [ ] Run `uv run --extra dev pytest tests/test_factor_attribution.py -q`.
- [ ] Run repository lint, format, typecheck, full test, and maintainability gates.
