from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from . import finance_risk as _native

__all__ = ["FactorRiskDecomposition", "factor_risk_decomposition"]


class FactorRiskDecomposition(BaseModel):
    model_config = ConfigDict(frozen=True)

    total_variance: float
    systematic_variance: float
    specific_variance: float
    factor_contributions: dict[str, float]


def factor_risk_decomposition(
    weights: Any,
    factor_loadings: Any,
    factor_covariance: Any,
    specific_variance: Any,
    *,
    factors: Sequence[str] | None = None,
) -> FactorRiskDecomposition:
    """Decompose portfolio variance into systematic factor and specific risk."""
    w = np.asarray(weights, dtype=float).reshape(-1)
    loadings = np.asarray(factor_loadings, dtype=float)
    factor_cov = np.asarray(factor_covariance, dtype=float)
    specific = np.asarray(specific_variance, dtype=float).reshape(-1)
    if loadings.shape[0] != w.size:
        raise ValueError("factor_loadings rows must match weights")
    if factor_cov.shape != (loadings.shape[1], loadings.shape[1]):
        raise ValueError("factor_covariance dimensions must match factor_loadings columns")
    if specific.size != w.size:
        raise ValueError("specific_variance length must match weights")
    factor_names = tuple(factors) if factors is not None else tuple(f"factor_{i}" for i in range(loadings.shape[1]))
    total, systematic, specific_var, contributions_raw = _native.factor_risk_decomposition(
        w.tolist(),
        loadings.tolist(),
        factor_cov.tolist(),
        specific.tolist(),
    )
    contributions = {name: float(contributions_raw[i]) for i, name in enumerate(factor_names)}
    return FactorRiskDecomposition(
        total_variance=float(total),
        systematic_variance=systematic,
        specific_variance=specific_var,
        factor_contributions=contributions,
    )
