# Factor return and risk attribution

This module provides a small multi-factor attribution primitive inspired by the useful parts of RQPAttr while keeping the platform's risk model, benchmark semantics, and cost accounting explicit.

## Active return

For portfolio weights `w`, benchmark weights `b`, exposures `X`, and factor returns `f`:

```text
active_weights   = w - b
active_exposure  = X' active_weights
factor_return    = active_exposure * f
net_active       = factor_return + specific_return - transaction_cost
```

`attribute_factor_return()` reports per-factor contributions, the residual specific return, and transaction-cost drag. The components reconcile to the supplied net active return.

## Active risk

For factor covariance `F` and per-asset specific risk `s`:

```text
factor variance contribution_i = e_i * (F e)_i
specific variance              = sum((active_weight_j * s_j)^2)
active variance                = sum(factor contribution) + specific variance
```

This assumes independent specific returns. It does not manufacture a factor covariance or specific-risk estimate; those are separate research inputs.

## Boundary

Brinson allocation/selection attribution is intentionally separate. A multi-factor risk model and Brinson industry allocation answer different questions and should not be mixed into one ambiguous decomposition.

A future Dashboard projection can display factor/style/industry contribution, specific contribution, and cost drag from these platform-owned results without recalculating them in React.
