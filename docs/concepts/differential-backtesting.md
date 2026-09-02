# Differential backtesting

External backtest frameworks are most useful here as independent references, not as new sources of truth. Every adapter first normalizes its output to `CanonicalBacktestResult`; `compare_backtest_results()` then compares that result against a chosen reference backend.

The current report localizes:

- shared numeric performance metrics by `period_end`;
- portfolio weight differences by `rebalance_date + symbol`;
- cash / positions value / NAV differences by `trade_date` when both backends expose a daily ledger;
- row-count differences for every canonical frame;
- capability differences, including declared market-rule coverage.

This is the groundwork for an RQAlpha A-share differential adapter. A future RQAlpha PR should use fixed fixtures and explain every remaining difference by market-rule semantics, fill timing, fees, cash accounting, or unsupported capabilities. Similar adapters can use the same report without adding framework-specific comparison code.

Orders and fills are currently summarized by row-count difference because canonical IDs are backend-local. A later execution-differential extension should define stable semantic matching keys before claiming fill-by-fill equivalence.
