import pandas as pd
import pytest


def test_matched_volatility_scales_to_common_risk_without_changing_sharpe():
    from portfolio_backtester.matched_volatility import match_realized_volatility

    a = pd.Series([0.01, -0.02, 0.03], index=pd.date_range("2020-01-01", periods=3))
    panel = pd.DataFrame({"a": a, "b": a * 0.5})
    result, factors = match_realized_volatility(panel)
    pd.testing.assert_series_equal(result.a, result.b, check_names=False)
    assert factors.to_dict() == pytest.approx({"a": 0.5, "b": 1.0})


def test_matched_volatility_rejects_missing_and_zero_volatility():
    from portfolio_backtester.matched_volatility import match_realized_volatility

    for values in ([0.0, 0.0, 0.0], [0.0, float("nan"), 0.1]):
        panel = pd.DataFrame({"a": values}, index=pd.date_range("2020-01-01", periods=3))
        with pytest.raises(ValueError):
            match_realized_volatility(panel)
