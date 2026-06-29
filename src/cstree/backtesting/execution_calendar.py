from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd

MARKET_CALENDAR = "market"
HK_CONNECT_CALENDAR = "hk_connect"

_MARKET_ALIASES = {
    "",
    "market",
    "hk_market",
    "trading",
    "trading_calendar",
    "data",
}
_HK_CONNECT_ALIASES = {
    "hk_connect",
    "stock_connect",
    "southbound",
    "southbound_connect",
    "connect",
}


def normalize_execution_calendar(value: object | None) -> str:
    text = str(value or MARKET_CALENDAR).strip().lower().replace("-", "_")
    if text in _MARKET_ALIASES:
        return MARKET_CALENDAR
    if text in _HK_CONNECT_ALIASES:
        return HK_CONNECT_CALENDAR
    raise ValueError("execution.calendar must be one of: market, hk_connect.")


def coerce_date_set(values: object | None) -> tuple[pd.Timestamp, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, pd.Timestamp)):
        raw_values: Iterable[object] = [values]
    elif isinstance(values, Iterable):
        raw_values = values
    else:
        raw_values = [values]

    dates: list[pd.Timestamp] = []
    for value in raw_values:
        text = str(value).strip()
        if not text:
            continue
        compact = text.replace("-", "")
        parsed = (
            pd.to_datetime(compact, format="%Y%m%d", errors="coerce")
            if compact.isdigit() and len(compact) == 8
            else pd.to_datetime(text, errors="coerce")
        )
        if pd.isna(parsed):
            raise ValueError(f"Invalid execution calendar date: {value}")
        dates.append(pd.Timestamp(parsed).normalize())
    return tuple(sorted(set(dates)))


def execution_calendar_from_config(cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    cfg = cfg if isinstance(cfg, Mapping) else {}
    return {
        "calendar": normalize_execution_calendar(cfg.get("calendar")),
        "open_dates": coerce_date_set(
            cfg.get("open_dates")
            or cfg.get("calendar_open_dates")
            or cfg.get("stock_connect_open_dates")
        ),
        "closed_dates": coerce_date_set(
            cfg.get("closed_dates")
            or cfg.get("calendar_closed_dates")
            or cfg.get("stock_connect_closed_dates")
        ),
    }


def _normalize_dates(values: Iterable[object] | None) -> list[pd.Timestamp]:
    if values is None:
        return []
    dates = [pd.Timestamp(value).normalize() for value in pd.to_datetime(list(values))]
    return sorted(set(dates))


def _date_range(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    if end < start:
        return []
    return list(pd.date_range(start, end, freq="D"))


def _weekday_dates(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    return [date for date in _date_range(start, end) if date.weekday() < 5]


def _fetch_trading_dates(
    rqdatac_module: Any,
    start: pd.Timestamp,
    end: pd.Timestamp,
    market_candidates: tuple[str, ...],
) -> list[pd.Timestamp] | None:
    start_text = start.strftime("%Y%m%d")
    end_text = end.strftime("%Y%m%d")
    for market in market_candidates:
        try:
            dates = rqdatac_module.get_trading_dates(start_text, end_text, market=market)
        except TypeError:
            try:
                dates = rqdatac_module.get_trading_dates(start_text, end_text)
            except Exception:
                continue
        except Exception:
            continue
        return _normalize_dates(dates)
    return None


def _resolve_rqdatac_module(rqdatac_module: Any | None) -> Any | None:
    if rqdatac_module is not None:
        return rqdatac_module
    try:
        import rqdatac  # type: ignore
    except ImportError:
        return None
    return rqdatac


def resolve_execution_open_dates(
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    calendar: str,
    base_dates: Iterable[object] | None = None,
    rqdatac_module: Any | None = None,
    open_dates: Iterable[object] | None = None,
    closed_dates: Iterable[object] | None = None,
    market: str = "hk",
) -> list[pd.Timestamp]:
    calendar = normalize_execution_calendar(calendar)
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    explicit_open = set(coerce_date_set(open_dates))
    explicit_closed = set(coerce_date_set(closed_dates))

    if explicit_open:
        candidates = [date for date in sorted(explicit_open) if start <= date <= end]
    elif calendar == MARKET_CALENDAR:
        module = _resolve_rqdatac_module(rqdatac_module)
        candidates = _normalize_dates(base_dates)
        if not candidates and module is not None:
            fetched = _fetch_trading_dates(module, start, end, (market, "hk"))
            candidates = fetched if fetched is not None else []
        if not candidates:
            candidates = _weekday_dates(start, end)
        candidates = [date for date in candidates if start <= date <= end]
    else:
        module = _resolve_rqdatac_module(rqdatac_module)
        hk_dates: list[pd.Timestamp] | None = None
        a_share_dates: list[pd.Timestamp] | None = None
        if module is not None:
            hk_dates = _fetch_trading_dates(module, start, end, ("hk", market))
            # Provider-specific RQData calendar aliases for A-share sessions.
            a_share_dates = _fetch_trading_dates(
                module,
                start,
                end,
                ("cn", "cn_stock", "stock_cn"),
            )
        if hk_dates is not None and a_share_dates is not None:
            candidates = sorted(set(hk_dates) & set(a_share_dates))
        else:
            candidates = _normalize_dates(base_dates)
            if not candidates:
                candidates = _weekday_dates(start, end)
            candidates = [date for date in candidates if start <= date <= end]

    return [date for date in sorted(set(candidates)) if date not in explicit_closed]


def is_execution_open(
    value: object,
    *,
    calendar: str,
    rqdatac_module: Any | None = None,
    open_dates: Iterable[object] | None = None,
    closed_dates: Iterable[object] | None = None,
    market: str = "hk",
) -> bool:
    date = pd.Timestamp(value).normalize()
    return date in set(
        resolve_execution_open_dates(
            date,
            date,
            calendar=calendar,
            rqdatac_module=rqdatac_module,
            open_dates=open_dates,
            closed_dates=closed_dates,
            market=market,
        )
    )


def resolve_execution_date(
    signal_date: object,
    shift_days: int,
    trade_dates: Iterable[object],
    *,
    calendar: str,
    rqdatac_module: Any | None = None,
    open_dates: Iterable[object] | None = None,
    closed_dates: Iterable[object] | None = None,
    market: str = "hk",
    allow_future: bool = False,
    horizon_days: int = 45,
) -> pd.Timestamp | None:
    calendar = normalize_execution_calendar(calendar)
    signal = pd.Timestamp(signal_date).normalize()
    normalized_trade_dates = _normalize_dates(trade_dates)
    end = max(normalized_trade_dates) if normalized_trade_dates else signal
    if allow_future:
        end = max(end, signal + pd.Timedelta(days=max(int(horizon_days), 7)))

    open_candidates = resolve_execution_open_dates(
        min(signal, min(normalized_trade_dates) if normalized_trade_dates else signal),
        end,
        calendar=calendar,
        base_dates=normalized_trade_dates if not allow_future else None,
        rqdatac_module=rqdatac_module,
        open_dates=open_dates,
        closed_dates=closed_dates,
        market=market,
    )
    trade_date_set = set(normalized_trade_dates)
    if not allow_future and trade_date_set:
        open_candidates = [date for date in open_candidates if date in trade_date_set]

    if shift_days <= 0:
        candidates = [date for date in open_candidates if date >= signal]
        offset = 0
    else:
        candidates = [date for date in open_candidates if date > signal]
        offset = int(shift_days) - 1
    if offset >= len(candidates):
        return None
    return pd.Timestamp(candidates[offset]).normalize()


def build_execution_date_map(
    rebalance_dates: Iterable[object],
    shift_days: int,
    trade_dates: Iterable[object],
    *,
    calendar: str,
    rqdatac_module: Any | None = None,
    open_dates: Iterable[object] | None = None,
    closed_dates: Iterable[object] | None = None,
    market: str = "hk",
    allow_future: bool = False,
) -> dict[pd.Timestamp, pd.Timestamp]:
    resolved: dict[pd.Timestamp, pd.Timestamp] = {}
    for rebalance_date in _normalize_dates(rebalance_dates):
        entry_date = resolve_execution_date(
            rebalance_date,
            shift_days,
            trade_dates,
            calendar=calendar,
            rqdatac_module=rqdatac_module,
            open_dates=open_dates,
            closed_dates=closed_dates,
            market=market,
            allow_future=allow_future,
        )
        if entry_date is not None:
            resolved[rebalance_date] = entry_date
    return resolved
