import numpy as np
import pandas as pd
import pytest


def test_volatility_decisions_use_only_observed_returns_and_keep_warmup():
    from portfolio_backtester.volatility_exposure import volatility_exposure

    returns = pd.Series([0.0, 0.1, -0.1, 0.1, -0.1], index=pd.date_range("2020-01-01", periods=5))
    result = volatility_exposure(returns, window=3, target_vol=0.15, floor=0.25)
    assert result.iloc[:2].tolist() == [1.0, 1.0]
    assert result.iloc[2] == pytest.approx(0.25)
    changed = returns.copy()
    changed.iloc[-1] = 10.0
    pd.testing.assert_series_equal(
        result.iloc[:-1],
        volatility_exposure(changed, window=3, target_vol=0.15, floor=0.25).iloc[:-1],
    )


def test_zero_volatility_is_full_exposure_and_missing_is_rejected():
    from portfolio_backtester.volatility_exposure import volatility_exposure

    returns = pd.Series([0.0] * 4, index=pd.date_range("2020-01-01", periods=4))
    assert volatility_exposure(returns, window=3).tolist() == [1.0] * 4
    returns.iloc[2] = np.nan
    with pytest.raises(ValueError):
        volatility_exposure(returns, window=3)
