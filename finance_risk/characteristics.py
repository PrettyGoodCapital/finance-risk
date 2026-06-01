from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

__all__ = [
    "CharacteristicFactorModel",
    "clean_characteristics",
    "fit_characteristic_factor_model",
]


class CharacteristicFactorModel(BaseModel):
    model_config = ConfigDict(frozen=True)

    factor_returns: tuple[float, ...]
    residual_variance: tuple[float, ...]


def clean_characteristics(values: Any, *, winsor_z: float = 3.0) -> np.ndarray:
    """Winsorize and z-score characteristic matrix column-wise."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2:
        raise ValueError("values must be a 2D array")

    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    std = np.where(std <= 1e-12, 1.0, std)
    z = (arr - mean) / std
    z = np.clip(z, -winsor_z, winsor_z)
    z_mean = np.nanmean(z, axis=0)
    z_std = np.nanstd(z, axis=0)
    z_std = np.where(z_std <= 1e-12, 1.0, z_std)
    return (z - z_mean) / z_std


def fit_characteristic_factor_model(
    returns: Any,
    characteristics: Any,
    *,
    ridge: float = 1e-6,
) -> CharacteristicFactorModel:
    """Fit cross-sectional returns on cleaned characteristics."""
    r = np.asarray(returns, dtype=float).reshape(-1)
    x = clean_characteristics(characteristics)
    if x.shape[0] != r.size:
        raise ValueError("characteristics rows must match returns length")

    xtx = x.T @ x + ridge * np.eye(x.shape[1])
    xty = x.T @ r
    beta = np.linalg.solve(xtx, xty)
    residuals = r - x @ beta
    return CharacteristicFactorModel(
        factor_returns=tuple(float(v) for v in beta),
        residual_variance=tuple(float(v) for v in residuals**2),
    )
