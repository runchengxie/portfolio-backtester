from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cstree.backtesting.metrics import summarize_ic


def test_summarize_ic_computes_finite_t_distribution_p_value() -> None:
    summary = summarize_ic(pd.Series([0.1, 0.2, 0.3, 0.4]))

    assert summary["n"] == 4
    assert summary["t_stat"] == pytest.approx(4.47213595499958)
    assert np.isfinite(summary["p_value"])
    assert summary["p_value"] == pytest.approx(0.020835151196184825)
