from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict

from . import finance_risk as _native

__all__ = [
    "CovarianceEstimate",
    "sample_covariance",
    "ledoit_wolf_covariance",
    "oas_covariance",
    "covariance_to_correlation",
    "correlation_to_covariance",
    "nearest_positive_semidefinite",
]


class CovarianceEstimate(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    matrix: np.ndarray
    assets: tuple[str, ...]
    method: str
    frequency: int = 252
    shrinkage: float | None = None


def _array(data: Any) -> np.ndarray:
    if hasattr(data, "to_numpy"):
        arr = data.to_numpy()
    else:
        arr = np.asarray(data)
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2:
        raise ValueError("returns must be a 2-D array with observations in rows and assets in columns")
    if arr.shape[0] < 2:
        raise ValueError("at least two observations are required")
    return arr


def _assets(data: Any, assets: Sequence[str] | None, width: int) -> tuple[str, ...]:
    if assets is not None:
        labels = tuple(str(asset) for asset in assets)
    elif hasattr(data, "columns"):
        labels = tuple(str(column) for column in data.columns)
    else:
        labels = tuple(f"asset_{i}" for i in range(width))
    if len(labels) != width:
        raise ValueError("assets length must match the number of columns")
    return labels


def _centered(data: Any) -> np.ndarray:
    arr = _array(data)
    mask = np.isfinite(arr).all(axis=1)
    arr = arr[mask]
    if arr.shape[0] < 2:
        raise ValueError("at least two complete observations are required")
    return arr - arr.mean(axis=0, keepdims=True)


def sample_covariance(data: Any, *, assets: Sequence[str] | None = None, frequency: int = 252) -> CovarianceEstimate:
    """Sample covariance of asset returns, annualized by ``frequency``."""
    arr = _array(data)
    labels = _assets(data, assets, arr.shape[1])
    matrix = np.asarray(_native.sample_covariance_matrix(arr.tolist(), float(frequency)), dtype=float)
    return CovarianceEstimate(matrix=nearest_positive_semidefinite(matrix), assets=labels, method="sample", frequency=frequency)


def ledoit_wolf_covariance(data: Any, *, assets: Sequence[str] | None = None, frequency: int = 252) -> CovarianceEstimate:
    """Ledoit-Wolf constant-variance shrinkage covariance estimate."""
    arr = _array(data)
    labels = _assets(data, assets, arr.shape[1])
    matrix_raw, shrinkage = _native.ledoit_wolf_covariance_matrix(arr.tolist(), float(frequency))
    matrix = np.asarray(matrix_raw, dtype=float)
    return CovarianceEstimate(
        matrix=nearest_positive_semidefinite(matrix), assets=labels, method="ledoit_wolf", frequency=frequency, shrinkage=shrinkage
    )


def oas_covariance(data: Any, *, assets: Sequence[str] | None = None, frequency: int = 252) -> CovarianceEstimate:
    """Oracle Approximating Shrinkage covariance estimate."""
    arr = _array(data)
    labels = _assets(data, assets, arr.shape[1])
    matrix_raw, shrinkage = _native.oas_covariance_matrix(arr.tolist(), float(frequency))
    matrix = np.asarray(matrix_raw, dtype=float)
    return CovarianceEstimate(
        matrix=nearest_positive_semidefinite(matrix), assets=labels, method="oas", frequency=frequency, shrinkage=float(shrinkage)
    )


def covariance_to_correlation(covariance: Any) -> np.ndarray:
    cov = np.asarray(covariance, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("covariance must be a square matrix")
    return np.asarray(_native.covariance_to_correlation(cov.tolist()), dtype=float)


def correlation_to_covariance(correlation: Any, volatility: Any) -> np.ndarray:
    corr = np.asarray(correlation, dtype=float)
    vol = np.asarray(volatility, dtype=float)
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
        raise ValueError("correlation must be a square matrix")
    if vol.shape[0] != corr.shape[0]:
        raise ValueError("volatility length must match correlation dimensions")
    return np.asarray(_native.correlation_to_covariance(corr.tolist(), vol.tolist()), dtype=float)


def nearest_positive_semidefinite(matrix: Any, *, epsilon: float = 0.0) -> np.ndarray:
    """Project a symmetric matrix onto the positive-semidefinite cone."""
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("matrix must be square")
    sym = (arr + arr.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals = np.clip(eigvals, epsilon, None)
    fixed = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return (fixed + fixed.T) / 2.0
