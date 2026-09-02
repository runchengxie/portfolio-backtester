"""DailyWatch20 construction: configuration, dataclasses, and cross-section prep."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

FallbackMode = Literal["none", "core20"]
ReceiptStatus = Literal["selected", "fallback", "unavailable"]

_OUTPUT_COLUMNS = {
    "blended_score",
    "dual_confirmed",
    "fallback_mode",
    "guard_prior",
    "ml_percentile",
    "retained_b",
    "selection_score",
    "sleeve",
    "sleeve_rank",
    "tracking_weight",
}


@dataclass(frozen=True)
class GuardFactorSpec:
    """One guard-prior component; factor values are ranked cross-sectionally."""

    column: str
    weight: float = 1.0
    higher_is_better: bool = True


@dataclass(frozen=True)
class DailyWatch20Config:
    """Construction settings for the strict four-name plus sixteen-name watchlist."""

    date_col: str = "trade_date"
    symbol_col: str = "symbol"
    industry_col: str = "first_industry_name"
    ml_score_col: str = "xgb_score"
    hard_eligibility_col: str = "hard_eligible"
    guard_factors: tuple[GuardFactorSpec, ...] = (GuardFactorSpec("guard_score"),)
    ml_weight: float = 0.60
    guard_weight: float = 0.40
    industry_cap: int = 4
    b_retention_buffer: int = 8
    b_max_replacements: int = 4
    a_tracking_weight: float = 0.20


@dataclass(frozen=True)
class DailyWatch20Receipt:
    """Auditable summary of one selection attempt."""

    status: ReceiptStatus
    trade_date: str | None
    fallback_mode: FallbackMode
    reason: str | None
    summary: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "trade_date": self.trade_date,
            "fallback_mode": self.fallback_mode,
            "reason": self.reason,
            **dict(self.summary),
        }


@dataclass(frozen=True)
class DailyWatch20Result:
    """Selected rows and the receipt that proves their construction."""

    watchlist: pd.DataFrame
    receipt: DailyWatch20Receipt
    candidate_scores: pd.DataFrame = field(default_factory=pd.DataFrame)


class DailyWatch20SelectionError(RuntimeError):
    """Fail-closed selection error carrying an unavailable receipt."""

    def __init__(self, message: str, receipt: DailyWatch20Receipt) -> None:
        super().__init__(message)
        self.receipt = receipt


@dataclass(frozen=True)
class _PreparedCrossSection:
    frame: pd.DataFrame
    trade_date: str
    input_summary: Mapping[str, int]


@dataclass(frozen=True)
class _BSelection:
    symbols: tuple[str, ...]
    retained: frozenset[str]
    previous_count: int
    exited_count: int
    added_count: int
    replacement_count: int
    forced_replacement_count: int


@dataclass(frozen=True)
class _SleeveSelection:
    a_symbols: tuple[str, ...]
    b_selection: _BSelection
    dual_confirmed: frozenset[str]


def _validate_config(config: DailyWatch20Config) -> None:
    text_fields = (
        config.date_col,
        config.symbol_col,
        config.industry_col,
        config.ml_score_col,
        config.hard_eligibility_col,
    )
    if any(not str(value).strip() for value in text_fields):
        raise ValueError("DailyWatch20 column names must be non-empty.")
    if not config.guard_factors:
        raise ValueError("DailyWatch20 requires at least one guard factor.")
    guard_columns = [factor.column for factor in config.guard_factors]
    if any(not str(column).strip() for column in guard_columns):
        raise ValueError("Guard factor column names must be non-empty.")
    if len(set(guard_columns)) != len(guard_columns):
        raise ValueError("Guard factor columns must be unique.")
    if any(not np.isfinite(factor.weight) or factor.weight <= 0 for factor in config.guard_factors):
        raise ValueError("Guard factor weights must be finite and positive.")
    if not np.isfinite(config.ml_weight) or not np.isfinite(config.guard_weight):
        raise ValueError("ML and guard weights must be finite.")
    if config.ml_weight < 0 or config.guard_weight < 0:
        raise ValueError("ML and guard weights must be non-negative.")
    if not np.isclose(config.ml_weight + config.guard_weight, 1.0):
        raise ValueError("ML and guard weights must sum to 1.")
    if config.industry_cap <= 0:
        raise ValueError("industry_cap must be positive.")
    if config.b_retention_buffer < 0 or config.b_max_replacements < 0:
        raise ValueError("B retention settings must be non-negative.")
    if not 0 <= config.a_tracking_weight <= 1:
        raise ValueError("a_tracking_weight must be between 0 and 1.")


def _required_columns(config: DailyWatch20Config) -> set[str]:
    return {
        config.date_col,
        config.symbol_col,
        config.industry_col,
        config.ml_score_col,
        config.hard_eligibility_col,
        *(factor.column for factor in config.guard_factors),
    }


def _truthy(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).ne(0)
    return series.astype("string").str.strip().str.lower().isin({"1", "true", "yes", "y"})


def _trade_date_text(value: object) -> str:
    parsed = pd.to_datetime(cast(Any, value), errors="coerce")
    if not pd.isna(parsed):
        return cast(pd.Timestamp, parsed).strftime("%Y-%m-%d")
    return str(value)


def _percentile_rank(values: pd.Series, *, higher_is_better: bool) -> pd.Series:
    return values.rank(
        method="average",
        ascending=higher_is_better,
        pct=True,
    )


def _prepare_cross_section(
    data: pd.DataFrame,
    config: DailyWatch20Config,
) -> _PreparedCrossSection:
    if data is None or data.empty:
        raise ValueError("DailyWatch20 input must be non-empty.")
    missing = sorted(_required_columns(config) - set(data.columns))
    if missing:
        raise ValueError(f"DailyWatch20 input missing required columns: {missing}")
    collisions = sorted(_OUTPUT_COLUMNS & set(data.columns))
    if collisions:
        raise ValueError(f"DailyWatch20 input contains reserved output columns: {collisions}")

    work = data.copy()
    dates = work[config.date_col].drop_duplicates()
    if len(dates) != 1 or bool(dates.isna().any()):
        raise ValueError("DailyWatch20 input must contain exactly one non-null trade date.")
    trade_date = _trade_date_text(dates.iloc[0])

    work[config.symbol_col] = work[config.symbol_col].astype("string").str.strip()
    symbols_valid = work[config.symbol_col].notna() & work[config.symbol_col].ne("")
    if not bool(symbols_valid.all()):
        raise ValueError("DailyWatch20 symbols must be non-empty.")
    if bool(work[config.symbol_col].duplicated().any()):
        raise ValueError("DailyWatch20 input must contain one row per symbol.")

    industry = work[config.industry_col].astype("string").str.strip()
    industry_valid = industry.notna() & industry.ne("")
    work[config.industry_col] = industry
    ml_score = pd.to_numeric(work[config.ml_score_col], errors="coerce")
    ml_valid = ml_score.notna() & np.isfinite(ml_score)
    hard_flag = _truthy(cast(pd.Series, work[config.hard_eligibility_col]))
    hard_eligible = hard_flag & industry_valid & ml_valid

    work["_ml_score"] = ml_score
    work["_hard_eligible"] = hard_eligible
    work["_ml_percentile"] = np.nan
    work.loc[hard_eligible, "_ml_percentile"] = _percentile_rank(
        ml_score.loc[hard_eligible], higher_is_better=True
    )

    guard_complete = hard_eligible.copy()
    numeric_guards: dict[str, pd.Series] = {}
    for factor in config.guard_factors:
        values = pd.to_numeric(work[factor.column], errors="coerce")
        numeric_guards[factor.column] = values
        guard_complete &= values.notna() & np.isfinite(values)

    factor_weight_sum = sum(factor.weight for factor in config.guard_factors)
    guard_prior = pd.Series(np.nan, index=work.index, dtype=float)
    if bool(guard_complete.any()):
        guard_prior.loc[guard_complete] = 0.0
        for factor in config.guard_factors:
            ranked = _percentile_rank(
                numeric_guards[factor.column].loc[guard_complete],
                higher_is_better=factor.higher_is_better,
            )
            guard_prior.loc[guard_complete] += ranked * factor.weight / factor_weight_sum

    work["_guard_prior"] = guard_prior
    work["_b_eligible"] = guard_complete
    work["_blended_score"] = (
        config.ml_weight * work["_ml_percentile"] + config.guard_weight * guard_prior
    )
    input_summary = {
        "input_rows": len(work),
        "hard_flag_true_count": int(hard_flag.sum()),
        "hard_eligible_count": int(hard_eligible.sum()),
        "b_eligible_count": int(guard_complete.sum()),
        "excluded_hard_flag_count": int((~hard_flag).sum()),
        "excluded_missing_ml_count": int((hard_flag & ~ml_valid).sum()),
        "excluded_missing_industry_count": int((hard_flag & ~industry_valid).sum()),
        "excluded_missing_guard_count": int((hard_eligible & ~guard_complete).sum()),
    }
    return _PreparedCrossSection(work, trade_date, input_summary)


def _base_receipt_summary(
    prepared: _PreparedCrossSection,
    config: DailyWatch20Config,
) -> dict[str, Any]:
    return {
        **dict(prepared.input_summary),
        "ml_weight": float(config.ml_weight),
        "guard_weight": float(config.guard_weight),
        "guard_factors": [
            {
                "column": factor.column,
                "weight": float(factor.weight),
                "higher_is_better": bool(factor.higher_is_better),
            }
            for factor in config.guard_factors
        ],
        "industry_cap": int(config.industry_cap),
        "b_retention_buffer": int(config.b_retention_buffer),
        "b_max_replacements": int(config.b_max_replacements),
    }


def _raise_unavailable(
    reason: str,
    *,
    prepared: _PreparedCrossSection,
    config: DailyWatch20Config,
    fallback_mode: FallbackMode,
    extra: Mapping[str, Any] | None = None,
) -> None:
    summary = _base_receipt_summary(prepared, config)
    summary.update(dict(extra or {}))
    receipt = DailyWatch20Receipt(
        status="unavailable",
        trade_date=prepared.trade_date,
        fallback_mode=fallback_mode,
        reason=reason,
        summary=summary,
    )
    raise DailyWatch20SelectionError(reason, receipt)
