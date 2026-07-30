"""Order construction submodules (split from orders.py for maintainability)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from ..execution import DetailedTradeFeeModel
from .capacity import (
    _capacity_notional,
    _capacity_weight,
    _execution_window_dates,
    _position_values_by_symbol,
    _price_at,
)
from .config import (
    SELL_UNTIL_NEXT_REBALANCE,
    ExecutionSimConfig,
)
from .models import (
    _ExecutionTables,
    _NavOrder,
    _OrderSink,
    _trade_fee,
)
from .reporting import (
    _format_date,
)

TradeFeeModel = DetailedTradeFeeModel

from .orders_targets import (
    _target_cash_notional,
    _cash_weight_breakdown,
    _cost_adjusted_target_notional,
    _build_targets_by_rebalance,
)
from .orders_ideal import (
    _rebalance_ideal_target,
    _build_ideal_rebalance_orders,
    _ideal_nav_order,
    _build_nav_orders_for_target,
    _nav_sell_max_days,
    _execute_ideal_sell_orders,
    _apply_ideal_sell_fill,
    _ideal_sell_status,
    _execute_ideal_buy_orders,
    _apply_ideal_buy_fill,
    _ideal_buy_status,
)
from .orders_nav import (
    _execute_sell_orders,
    _execute_buy_orders,
    _execute_nav_orders_for_day,
    _execute_nav_sell_orders_for_day,
    _execute_nav_buy_orders_for_day,
    _nav_order_is_complete,
    _update_nav_order,
    _record_nav_fill,
    _nav_order_should_abort_buy,
    _finalize_open_nav_orders,
    _append_nav_order_row,
    _build_order_states,
    _update_state,
    _append_order_rows,
    _record_fill,
)

__all__ = [
    '_target_cash_notional',
    '_cash_weight_breakdown',
    '_cost_adjusted_target_notional',
    '_build_targets_by_rebalance',
    '_rebalance_ideal_target',
    '_build_ideal_rebalance_orders',
    '_ideal_nav_order',
    '_build_nav_orders_for_target',
    '_nav_sell_max_days',
    '_execute_ideal_sell_orders',
    '_apply_ideal_sell_fill',
    '_ideal_sell_status',
    '_execute_ideal_buy_orders',
    '_apply_ideal_buy_fill',
    '_ideal_buy_status',
    '_execute_sell_orders',
    '_execute_buy_orders',
    '_execute_nav_orders_for_day',
    '_execute_nav_sell_orders_for_day',
    '_execute_nav_buy_orders_for_day',
    '_nav_order_is_complete',
    '_update_nav_order',
    '_record_nav_fill',
    '_nav_order_should_abort_buy',
    '_finalize_open_nav_orders',
    '_append_nav_order_row',
    '_build_order_states',
    '_update_state',
    '_append_order_rows',
    '_record_fill',
]
