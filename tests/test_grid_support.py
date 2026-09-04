import pandas as pd

from portfolio_backtester.grid_support import (
    GRID_RESULT_FIELDS,
    init_grid_row,
    parse_date_list,
    resolve_rebalance_dates,
    safe_run_name,
    write_grid_rows,
)


def test_parse_date_list_sorts_and_deduplicates():
    assert parse_date_list(["20200103", "2020-01-02", "20200103", ""]) == [
        pd.Timestamp("2020-01-02"),
        pd.Timestamp("2020-01-03"),
    ]


def test_safe_run_name_is_stable():
    assert (
        safe_run_name(
            "demo",
            20,
            12.5,
            buffer_exit=2,
            buffer_entry=3,
            include_buffer=True,
            weighting="equal",
            include_weighting=True,
        )
        == "demo_k20_bps12p5_bx2_be3_wequal"
    )


def test_resolve_rebalance_dates_filters_small_dates():
    scored = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2020-01-02", "2020-01-02", "2020-01-03"]),
            "symbol": ["A", "B", "A"],
        }
    )
    dates = resolve_rebalance_dates(None, scored, "D", min_symbols_per_date=2)
    assert dates == [pd.Timestamp("2020-01-02")]


def test_grid_result_helpers_create_stable_csv(tmp_path):
    row = init_grid_row(20, 12.5, 2, 3, "equal", "demo", tmp_path / "summary.json")
    output = tmp_path / "reports" / "grid.csv"
    write_grid_rows(output, [row])
    assert row["status"] == "ok"
    assert output.read_text(encoding="utf-8").splitlines()[0].split(",") == list(
        GRID_RESULT_FIELDS
    )
