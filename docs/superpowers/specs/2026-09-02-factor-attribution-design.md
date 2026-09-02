# Factor Attribution Design

## Goal

Add a framework-neutral active return and active risk attribution primitive that can consume a platform risk model and later feed Dashboard evidence.

## Design

Return attribution decomposes net active return into factor contributions, a residual specific term, and explicit transaction-cost drag. Risk attribution decomposes active variance into per-factor variance contributions plus independent specific variance.

The module accepts weights, benchmark weights, exposures, factor returns/covariance, and specific risk as plain pandas objects. It does not estimate the risk model, choose a benchmark, or implement Brinson allocation/selection attribution.

## Non-goals

- no proprietary RQPAttr dependency;
- no blending of Brinson and multifactor attribution semantics;
- no estimation of factor covariance/specific risk;
- no Dashboard-side recomputation.
