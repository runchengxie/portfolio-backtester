"""Input-frame normalization for target-position construction."""

from __future__ import annotations

import pandas as pd

from portfolio_backtester._symbol_utils import canonicalize_symbol_columns


def normalize_portfolio_frames(
    data: pd.DataFrame,
    pricing_data: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    if data is not None and not data.empty:
        data = canonicalize_symbol_columns(data, context="Portfolio data")
        data = data.copy()
        data["trade_date"] = pd.to_datetime(data["trade_date"]).dt.normalize()
    if pricing_data is not None and not pricing_data.empty:
        pricing_data = canonicalize_symbol_columns(
            pricing_data,
            context="Portfolio pricing data",
        )
        pricing_data = pricing_data.copy()
        pricing_data["trade_date"] = pd.to_datetime(pricing_data["trade_date"]).dt.normalize()
    return data, pricing_data


def resolve_pricing_source(
    data: pd.DataFrame,
    pricing_data: pd.DataFrame | None,
) -> pd.DataFrame | None:
    if pricing_data is not None and not pricing_data.empty:
        return pricing_data
    return data
