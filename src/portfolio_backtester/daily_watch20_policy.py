"""Owner-native construction and execution policy for DailyWatch20."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

PORTFOLIO_POLICY_SCHEMA = "daily_watch20.portfolio_policy.v1"


@dataclass(frozen=True, slots=True)
class DailyWatch20PortfolioPolicy:
    portfolio_size: int = 20
    weighting: str = "equal_weight"
    single_side_cost_bps: float = 20.0
    execution_horizons: tuple[int, ...] = (1, 3, 5)
    tradability_price: str = "raw_open"
    valuation_price: str = "adj_open"
    blocked_buy_policy: str = "keep_cash_without_redistribution"
    blocked_sell_policy: str = "carry_and_retry_next_open"
    terminal_policy: str = "fail_closed"

    def __post_init__(self) -> None:
        if self.portfolio_size <= 0:
            raise ValueError("portfolio_size must be positive")
        if self.weighting != "equal_weight":
            raise ValueError("DailyWatch20 weighting is frozen at equal_weight")
        if not 0 <= self.single_side_cost_bps < 10_000:
            raise ValueError("single_side_cost_bps must be in [0, 10000)")
        if not self.execution_horizons or any(
            value not in {1, 3, 5} for value in self.execution_horizons
        ):
            raise ValueError("execution_horizons must use the supported 1/3/5-day grid")
        if len(self.execution_horizons) != len(set(self.execution_horizons)):
            raise ValueError("execution_horizons must be unique")
        expected = {
            "tradability_price": "raw_open",
            "blocked_buy_policy": "keep_cash_without_redistribution",
            "blocked_sell_policy": "carry_and_retry_next_open",
            "terminal_policy": "fail_closed",
        }
        changed = [name for name, value in expected.items() if getattr(self, name) != value]
        if changed:
            raise ValueError(f"portfolio execution safety policy is frozen: {', '.join(changed)}")

    @property
    def policy_id(self) -> str:
        payload = {"schema_version": PORTFOLIO_POLICY_SCHEMA, **asdict(self)}
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16]
        return f"{PORTFOLIO_POLICY_SCHEMA}:{digest}"

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": PORTFOLIO_POLICY_SCHEMA,
            "policy_id": self.policy_id,
            **asdict(self),
        }


__all__ = ["PORTFOLIO_POLICY_SCHEMA", "DailyWatch20PortfolioPolicy"]
