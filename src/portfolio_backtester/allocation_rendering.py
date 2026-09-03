from __future__ import annotations

import io
import math
import unicodedata
from typing import SupportsFloat, cast

import pandas as pd


def safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        result = float(cast(SupportsFloat, value))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def display_width(text: str) -> int:
    width = 0
    for char in text:
        if unicodedata.combining(char):
            continue
        if unicodedata.east_asian_width(char) in {"F", "W"}:
            width += 2
        else:
            width += 1
    return width


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [display_width(header) for header in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], display_width(value))

    def format_row(row: list[str]) -> str:
        cells: list[str] = []
        for idx, value in enumerate(row):
            padding = widths[idx] - display_width(value)
            cells.append(f" {value}{' ' * padding} ")
        return "|" + "|".join(cells) + "|"

    separator = "|" + "|".join("-" * (width + 2) for width in widths) + "|"
    return "\n".join(
        [
            format_row(headers),
            separator,
            *(format_row(row) for row in rows),
        ]
    )


def format_allocation_text(payload: dict, allocations: pd.DataFrame) -> str:
    rows: list[list[str]] = []
    for _, row in allocations.iterrows():
        rows.append(
            [
                str(row["symbol"]),
                str(row["order_book_id"]),
                str(row["side"]),
                f"{safe_float(row['price']) or 0.0:.4f}",
                str(int(row["round_lot"])),
                str(int(row["shares"])),
                f"{safe_float(row['est_value']) or 0.0:.2f}",
                f"{safe_float(row['gap_to_target']) or 0.0:.2f}",
            ]
        )
    table = format_table(
        ["symbol", "order_book_id", "side", "price", "lot", "shares", "est_value", "gap"],
        rows,
    )
    return (
        f"Entry date: {payload['entry_date']} | Price date: {payload['price_date']} | "
        f"Source: {payload['source']} | Side: {payload['side']}\n"
        f"Reference: {payload['reference_source']} | File: {payload['reference_file']}\n"
        f"Cash: {payload['cash']:.2f} | Buffer: {payload['buffer_bps']:.2f} bps | "
        f"Investable: {payload['investable_cash']:.2f}\n"
        f"Estimated value: {payload['estimated_value']:.2f} | "
        f"Cash left: {payload['cash_left']:.2f} | "
        f"Gap: {payload['total_gap_to_target']:.2f}\n"
        f"{table}"
    )


def format_allocation_csv(allocations: pd.DataFrame) -> str:
    buffer = io.StringIO()
    allocations.to_csv(buffer, index=False)
    return buffer.getvalue().rstrip("\n")
