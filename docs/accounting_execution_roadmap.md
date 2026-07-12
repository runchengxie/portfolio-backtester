# Backtest accounting and execution roadmap

This document records the staged migration toward one auditable accounting path for signal,
position, and capacity backtests. It is intentionally split into reviewable changes rather
than one repository-wide rewrite.

## Invariants

Every backtest path should eventually satisfy the same invariants:

1. Target weights generate signed orders.
2. Orders generate zero or more fills.
3. Fills update shares and cash.
4. Daily NAV equals cash plus marked position value.
5. Explicit fees and implicit execution costs are reported separately.
6. Turnover fields have one documented formula across every entry point.
7. Missing configured execution inputs fail closed instead of silently disabling constraints.

## Phase 1: terminology and result contracts

Implemented by the first accounting PR:

- distinguish holding-name replacement from weight turnover;
- expose buy, sell, gross traded, half-L1, and one-way turnover fields;
- preserve the historical initial-build cost convention without hiding half-L1 turnover;
- expose a typed cost breakdown on leg and period results;
- document the accounting formulas in the public README.

## Phase 2: shared daily ledger

Move `backtest_topk`, `run_position_backtest`, ideal NAV, and capacity-adjusted NAV onto a
shared ledger core:

```text
Targets -> Orders -> Fills -> Shares/Cash -> Daily NAV -> Reports
```

The signal backtester should stop maintaining its own return-and-cost arithmetic. It should
construct targets and delegate accounting to the same engine used by externally supplied
positions.

Required outputs:

- `targets`;
- `orders`;
- `fills`;
- `daily_positions`;
- `daily_cash`;
- `daily_nav`;
- `cost_breakdown`;
- `turnover_breakdown`.

## Phase 3: cost decomposition

Replace the overloaded detailed fee model with components that cannot be double counted:

- commission;
- stamp duty and exchange/transfer fees;
- quoted-spread or half-spread cost;
- temporary market impact;
- permanent impact, when calibrated;
- opportunity cost from delayed or abandoned orders;
- borrow and financing costs.

Minimum commission must declare its charging unit. The recommended default is one charge per
symbol, side, and trading day, with an optional broker-specific policy.

## Phase 4: market rules and timestamps

Add a market rule contract for:

- buy lot size and odd-lot liquidation;
- T+1 sellability;
- price limits and direction-specific tradability;
- listing, suspension, and delisting behavior;
- fee schedules with effective dates.

Execution and valuation timestamps should be explicit:

- signal time;
- decision time;
- order submission time;
- fill time;
- mark time.

This enables automated look-ahead checks and separates execution price from mark price.

## Phase 5: capacity and impact calibration

Capacity constraints should limit fills. They should not merely cap the participation value
used inside a cost formula. Exceeding a participation limit must create an unfilled remainder
or an explicitly labelled extrapolation.

Capacity reports should add:

- break-even AUM;
- AUM at 95% fill ratio;
- AUM at a configurable alpha-retention threshold;
- marginal impact per additional capital increment;
- concentration of capacity usage by symbol, industry, and liquidity bucket.

Execution-window liquidity should be preferred over full-day amount when the strategy trades
inside a narrower window.

## Phase 6: metrics and reproducibility

All headline return and risk statistics should be derived from daily NAV. Calendar summaries
must compound returns inside each calendar period and retain the year dimension.

Each run should persist:

- repository commit;
- configuration hash;
- input-data hashes;
- universe and calendar versions;
- fee schedule and slippage calibration versions;
- package versions;
- random seed;
- run timestamp.

## Test strategy

The ledger migration should be guarded by:

- cash and position-value conservation tests;
- zero-return, zero-cost NAV invariants;
- cost monotonicity tests;
- capacity monotonicity tests;
- replay equivalence between Top-K targets and position backtests;
- order-level minimum-commission tests;
- golden ledgers with manually verified daily cash, shares, and NAV;
- property-based tests over sparse weights, missing prices, and delayed fills.
