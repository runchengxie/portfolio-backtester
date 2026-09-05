import pandas as pd
import pytest


def test_episodes_include_recovered_and_censored_drawdowns():
    from portfolio_backtester.drawdown_episodes import drawdown_episodes

    dates = pd.date_range("2020-01-01", periods=5)
    result = drawdown_episodes(pd.Series([100.0, 90.0, 80.0, 100.0, 95.0], index=dates))
    assert result.depth.tolist() == pytest.approx([-0.2, -0.05])
    assert result.underwater_sessions.tolist() == [2, 1]
    assert result.elapsed_calendar_days.tolist() == [3, 1]
    assert result.censored.tolist() == [False, True]
    assert result.peak_date.tolist() == [dates[0], dates[3]]
    assert result.recovery_date.iloc[0] == dates[3]
    assert pd.isna(result.recovery_date.iloc[1])
    assert result.trough_date.tolist() == [dates[2], dates[4]]


def test_flat_nav_has_no_underwater_episodes():
    from portfolio_backtester.drawdown_episodes import drawdown_episodes

    result = drawdown_episodes(
        pd.Series([1.0, 1.0, 1.0], index=pd.date_range("2020-01-01", periods=3))
    )
    assert result.empty
    assert "censored" in result.columns


@pytest.mark.parametrize("values", [[1.0, 0.0], [1.0, float("nan")], [1.0, float("inf")]])
def test_invalid_nav_is_not_silently_removed(values):
    from portfolio_backtester.drawdown_episodes import drawdown_episodes

    with pytest.raises(ValueError):
        drawdown_episodes(pd.Series(values, index=pd.date_range("2020-01-01", periods=2)))


def test_duplicate_and_unsorted_dates_are_rejected():
    from portfolio_backtester.drawdown_episodes import drawdown_episodes

    for dates in (["2020-01-01", "2020-01-01"], ["2020-01-02", "2020-01-01"]):
        with pytest.raises(ValueError):
            drawdown_episodes(pd.Series([1.0, 0.9], index=pd.to_datetime(dates)))
