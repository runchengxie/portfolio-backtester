# Differential Backtest Design

## Goal

Provide framework-neutral evidence for comparing a native backtest with an external backend such as RQAlpha.

## Design

Both backends must first emit validated `CanonicalBacktestResult` objects. The differential layer compares shared numeric performance metrics, weights, daily ledger accounting, frame row counts, and capability declarations. It returns localized dataframes plus a machine-readable summary.

Backend-local order/fill identifiers are not treated as semantic keys. This first version reports their row-count deltas only; fill-level equivalence requires a later explicit semantic matching contract.

## Non-goals

- no RQAlpha dependency in this PR;
- no external framework object in the report;
- no tolerance-based declaration that two strategies are economically equivalent;
- no bypass of A-share native market-rule ownership.
