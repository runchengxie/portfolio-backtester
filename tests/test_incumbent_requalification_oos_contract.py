from __future__ import annotations

import pandas as pd
import pytest

from portfolio_backtester.incumbent_requalification import (
    IncumbentRequalificationConfig,
    IncumbentRequalificationPolicy,
)
from portfolio_backtester.incumbent_requalification_oos import (
    stateful_incumbent_requalification_daily_rows,
    stateless_incumbent_requalification_daily_rows,
)


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.to_datetime(["2026-07-16", "2026-07-17"])
    scores = {
        dates[0]: {
            "A": (1.00, True, True),
            "B": (0.90, True, True),
            "C": (0.80, True, True),
            "D": (0.70, True, True),
            "E": (0.60, True, True),
        },
        dates[1]: {
            "D": (1.00, True, True),
            "B": (0.90, True, True),
            "E": (0.85, True, True),
            "A": (0.80, True, False),
            "C": (0.70, False, False),
        },
    }
    scored_rows: list[dict[str, object]] = []
    frame_rows: list[dict[str, object]] = []
    for date, values in scores.items():
        execution_date = date + pd.offsets.BDay(1)
        for index, (symbol, (score, hard, entry)) in enumerate(values.items()):
            scored_rows.append(
                {
                    "trade_date": date,
                    "symbol": symbol,
                    "relative_percentile": score,
                    "industry": f"industry_{index}",
                    "hard_eligible": hard,
                    "entry_eligible": entry,
                    "forward_return_1d": 0.001 * (index + 1),
                    "forward_label_start_date": execution_date,
                }
            )
            frame_rows.append(
                {
                    "trade_date": execution_date,
                    "symbol": symbol,
                    "open": 10.0,
                    "up_limit": 11.0,
                    "down_limit": 9.0,
                    "is_suspended": False,
                }
            )
    return pd.DataFrame(scored_rows), pd.DataFrame(frame_rows)


def _policy() -> IncumbentRequalificationPolicy:
    return IncumbentRequalificationPolicy(
        portfolio_size=3,
        entry_rank_limit=3,
        exit_rank_limit=4,
        max_new_positions=1,
        industry_cap=3,
    )


def test_stateful_control_isolates_incumbent_buffer() -> None:
    scored, frame = _inputs()

    stateful = stateful_incumbent_requalification_daily_rows(
        scored,
        frame,
        policy=_policy(),
        single_side_cost_bps=10.0,
    )
    stateless = stateless_incumbent_requalification_daily_rows(
        scored,
        frame,
        policy=_policy(),
        single_side_cost_bps=10.0,
    )

    assert stateful.loc[1, "selected_symbols"] == "A|B|D"
    assert stateless.loc[1, "selected_symbols"] == "B|D|E"
    assert stateful.loc[1, "buffered_incumbent_count"] == 1
    assert stateless.loc[1, "buffered_incumbent_count"] == 0
    assert stateful.loc[1, "one_way_turnover"] < stateless.loc[1, "one_way_turnover"]
    assert set(stateful["selection_state_mode"]) == {"carry"}
    assert set(stateless["selection_state_mode"]) == {"reset"}


def test_bridge_requires_explicit_entry_eligibility() -> None:
    scored, frame = _inputs()

    with pytest.raises(ValueError, match="entry_eligible"):
        stateful_incumbent_requalification_daily_rows(
            scored.drop(columns="entry_eligible"),
            frame,
            policy=_policy(),
            single_side_cost_bps=10.0,
        )


def test_bridge_supports_custom_selection_columns() -> None:
    scored, frame = _inputs()
    custom = scored.rename(
        columns={
            "trade_date": "date",
            "symbol": "ticker",
            "relative_percentile": "score",
            "industry": "sector",
            "hard_eligible": "hard",
            "entry_eligible": "can_enter",
        }
    )
    config = IncumbentRequalificationConfig(
        date_col="date",
        symbol_col="ticker",
        score_col="selection_score",
        industry_col="sector",
        hard_eligibility_col="hard",
        entry_eligibility_col="can_enter",
    )

    result = stateful_incumbent_requalification_daily_rows(
        custom,
        frame,
        policy=_policy(),
        single_side_cost_bps=10.0,
        column_config=config,
        score_column="score",
    )

    assert len(result) == 2
    assert result.loc[1, "selected_symbols"] == "A|B|D"
