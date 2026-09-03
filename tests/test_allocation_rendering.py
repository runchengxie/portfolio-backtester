from __future__ import annotations

import unicodedata

from portfolio_backtester.allocation_rendering import format_table


def _display_width(text: str) -> int:
    width = 0
    for char in text:
        if unicodedata.combining(char):
            continue
        width += 2 if unicodedata.east_asian_width(char) in {"F", "W"} else 1
    return width


def test_format_table_keeps_alignment_with_cjk_headers() -> None:
    table = format_table(
        ["symbol", "手数", "价格"],
        [["00941.HK", "2", "80.4000"]],
    )
    lines = table.splitlines()
    assert len(lines) == 3
    widths = [_display_width(line) for line in lines]
    assert widths[0] == widths[1] == widths[2]
