from pathlib import Path


def test_portfolio_package_does_not_own_freshness_score_transform() -> None:
    duplicate = Path("src/portfolio_backtester/freshness_overlay.py")
    assert not duplicate.exists(), (
        "freshness score transformation is alpha-owned; portfolio-backtester "
        "must consume an already-transformed score"
    )
