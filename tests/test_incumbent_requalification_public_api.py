from __future__ import annotations


def test_incumbent_requalification_is_exported_from_package_root() -> None:
    from portfolio_backtester import (
        INCUMBENT_REQUALIFICATION_SCHEMA,
        IncumbentRequalificationConfig,
        IncumbentRequalificationPolicy,
        IncumbentRequalificationReceipt,
        IncumbentRequalificationResult,
        select_incumbent_requalified_portfolio,
    )

    assert INCUMBENT_REQUALIFICATION_SCHEMA.endswith(".v1")
    assert IncumbentRequalificationConfig is not None
    assert IncumbentRequalificationPolicy is not None
    assert IncumbentRequalificationReceipt is not None
    assert IncumbentRequalificationResult is not None
    assert callable(select_incumbent_requalified_portfolio)
