from __future__ import annotations

import pandas as pd
import pytest

from portfolio_backtester.liquidity_proxy import (
    derive_execution_liquidity_proxy_columns,
    parse_execution_liquidity_proxy_column,
)


def test_parse_execution_liquidity_proxy_column() -> None:
    assert parse_execution_liquidity_proxy_column("adv20_amount") == ("adv", 20)
    assert parse_execution_liquidity_proxy_column("medadv5_amount") == ("medadv", 5)
    assert parse_execution_liquidity_proxy_column("adv0_amount") is None
    assert parse_execution_liquidity_proxy_column("amount") is None


def test_derive_execution_liquidity_proxy_columns_uses_lagged_amount() -> None:
    frame = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-01",
                    "2024-01-02",
                ]
            ),
            "symbol": ["AAA", "AAA", "AAA", "BBB", "BBB"],
            "amount": [10.0, 20.0, 40.0, 100.0, 200.0],
        }
    )

    out = derive_execution_liquidity_proxy_columns(
        frame.copy(),
        {"adv2_amount", "medadv2_amount"},
    )

    assert pd.isna(out["adv2_amount"].iloc[0])
    assert out["adv2_amount"].iloc[1:].tolist() == pytest.approx(
        [10.0, 15.0, float("nan"), 100.0], nan_ok=True
    )
    assert pd.isna(out["medadv2_amount"].iloc[0])
    assert out["medadv2_amount"].iloc[1:].tolist() == pytest.approx(
        [10.0, 15.0, float("nan"), 100.0],
        nan_ok=True,
    )


def test_derive_execution_liquidity_proxy_columns_keeps_existing_column() -> None:
    frame = pd.DataFrame(
        {
            "trade_date": ["2024-01-01"],
            "symbol": ["AAA"],
            "amount": [10.0],
            "adv2_amount": [99.0],
        }
    )

    out = derive_execution_liquidity_proxy_columns(frame.copy(), {"adv2_amount"})

    assert out["adv2_amount"].tolist() == [99.0]
