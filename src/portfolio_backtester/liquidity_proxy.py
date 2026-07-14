from __future__ import annotations

import logging
import re

import pandas as pd

logger = logging.getLogger("portfolio_backtester")
EXECUTION_LIQUIDITY_PROXY_PATTERN = re.compile(r"^(?P<kind>adv|medadv)(?P<window>\d+)_amount$")


def parse_execution_liquidity_proxy_column(column: str) -> tuple[str, int] | None:
    match = EXECUTION_LIQUIDITY_PROXY_PATTERN.fullmatch(str(column).strip())
    if match is None:
        return None
    window = int(match.group("window"))
    if window <= 0:
        return None
    return match.group("kind"), window


def derive_execution_liquidity_proxy_columns(
    df: pd.DataFrame,
    required_columns: set[str],
) -> pd.DataFrame:
    if df.empty or "amount" not in df.columns:
        return df
    out = df
    derived_specs: list[tuple[str, str, int]] = []
    for column in sorted(required_columns):
        if column in out.columns:
            continue
        parsed = parse_execution_liquidity_proxy_column(column)
        if parsed is None:
            continue
        kind, window = parsed
        derived_specs.append((column, kind, window))

    if not derived_specs:
        return out
    derived_columns: list[str] = []
    amount = pd.to_numeric(out["amount"], errors="coerce")
    lagged = amount.groupby(out["symbol"], sort=False).shift(1)
    grouped_lagged = lagged.groupby(out["symbol"], sort=False)
    for column, kind, window in derived_specs:
        if kind == "adv":
            proxy = grouped_lagged.transform(
                lambda series, window=window: series.rolling(window=window, min_periods=1).mean()
            )
        else:
            proxy = grouped_lagged.transform(
                lambda series, window=window: series.rolling(window=window, min_periods=1).median()
            )

        out[column] = pd.to_numeric(proxy, errors="coerce")
        derived_columns.append(column)
    if derived_columns:
        logger.info(
            "Derived execution liquidity proxy columns from lagged amount: %s",
            derived_columns,
        )
    return out


_parse_execution_liquidity_proxy_column = parse_execution_liquidity_proxy_column
_derive_execution_liquidity_proxy_columns = derive_execution_liquidity_proxy_columns


__all__ = [
    "_derive_execution_liquidity_proxy_columns",
    "_parse_execution_liquidity_proxy_column",
    "derive_execution_liquidity_proxy_columns",
    "parse_execution_liquidity_proxy_column",
]
