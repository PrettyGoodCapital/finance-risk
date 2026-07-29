from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from . import finance_risk as _native

__all__ = ["CointegrationSpread", "cointegration_spread", "rolling_correlation", "transfer_entropy"]


class CointegrationSpread(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    hedge_ratio: float
    intercept: float
    spread: np.ndarray


def _vector(values: Any) -> np.ndarray:
    if hasattr(values, "to_numpy"):
        arr = values.to_numpy()
    else:
        arr = np.asarray(values)
    arr = np.asarray(arr, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError("values must not be empty")
    return arr


def rolling_correlation(x: Any, y: Any, *, window: int) -> np.ndarray:
    """Rolling Pearson correlation with leading ``nan`` values before the first full window."""
    left = _vector(x)
    right = _vector(y)
    if left.shape != right.shape:
        raise ValueError("x and y must have the same length")
    return np.asarray(_native.rolling_correlation(left.tolist(), right.tolist(), window), dtype=float)


def cointegration_spread(y: Any, x: Any) -> CointegrationSpread:
    """OLS hedge ratio and demeaned residual spread for a two-asset pair."""
    dependent = _vector(y)
    independent = _vector(x)
    if dependent.shape != independent.shape:
        raise ValueError("x and y must have the same length")
    hedge_ratio, intercept, spread = _native.cointegration_spread(dependent.tolist(), independent.tolist())
    return CointegrationSpread(hedge_ratio=float(hedge_ratio), intercept=float(intercept), spread=np.asarray(spread, dtype=float))


def transfer_entropy(source: Any, target: Any, *, bins: int = 3, lag: int = 1) -> float:
    """Discrete transfer entropy ``source[t-lag] -> target[t]`` in nats."""
    src = _vector(source)
    tgt = _vector(target)
    if src.shape != tgt.shape:
        raise ValueError("source and target must have the same length")
    return float(_native.transfer_entropy(src.tolist(), tgt.tolist(), bins, lag))
