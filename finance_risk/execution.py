from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from . import finance_risk as _native

__all__ = [
    "ExecutionTrajectory",
    "MarketImpactEstimate",
    "almgren_chriss_market_impact",
    "optimal_execution_trajectory",
    "twap_schedule",
    "vwap_schedule",
]


class MarketImpactEstimate(BaseModel):
    model_config = ConfigDict(frozen=True)

    spread_cost: float
    temporary_impact: float
    permanent_impact: float
    total_cost: float


class ExecutionTrajectory(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    remaining: np.ndarray
    trades: np.ndarray


def twap_schedule(*, total_quantity: float, periods: int) -> np.ndarray:
    return np.asarray(_native.twap_schedule(float(total_quantity), periods), dtype=float)


def vwap_schedule(*, total_quantity: float, volume_curve: Any) -> np.ndarray:
    curve = np.asarray(volume_curve, dtype=float).reshape(-1)
    return np.asarray(_native.vwap_schedule(float(total_quantity), curve.tolist()), dtype=float)


def almgren_chriss_market_impact(
    *,
    quantity: float,
    volatility: float,
    daily_volume: float,
    spread: float = 0.0,
    temporary_impact: float = 0.1,
    permanent_impact: float = 0.1,
) -> MarketImpactEstimate:
    """Simple Almgren-Chriss style spread, temporary, and permanent cost estimate."""
    spread_cost, temp_cost, perm_cost, total_cost = _native.almgren_chriss_market_impact(
        float(quantity), float(volatility), float(daily_volume), float(spread), float(temporary_impact), float(permanent_impact)
    )
    return MarketImpactEstimate(
        spread_cost=spread_cost,
        temporary_impact=temp_cost,
        permanent_impact=perm_cost,
        total_cost=total_cost,
    )


def optimal_execution_trajectory(*, total_quantity: float, periods: int, risk_aversion: float = 1.0) -> ExecutionTrajectory:
    """Monotone liquidation trajectory with risk-aversion controlled front-loading."""
    remaining_raw, trades_raw = _native.optimal_execution_trajectory(float(total_quantity), periods, float(risk_aversion))
    remaining = np.asarray(remaining_raw, dtype=float)
    trades = np.asarray(trades_raw, dtype=float)
    return ExecutionTrajectory(remaining=remaining, trades=trades)
