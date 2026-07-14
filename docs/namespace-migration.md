# Portfolio namespace migration

- Canonical package: `portfolio_backtester.*`
- Deprecated package: `cstree.backtesting.*`
- Compatibility owner: `strategy-pipeline`
- Compatibility window: portfolio-backtester 0.2 / workspace 1.x
- Removal target: workspace 2.0

This distribution no longer installs `cstree` or uses `pkgutil.extend_path`.
