from __future__ import annotations

import numpy as np
import pytest

import finance_risk as fr


def test_clean_characteristics_standardizes_columns() -> None:
    raw = np.array(
        [
            [10.0, 2.0],
            [11.0, 2.1],
            [9.5, 1.8],
            [100.0, 2.2],
        ]
    )

    clean = fr.clean_characteristics(raw, winsor_z=2.5)

    assert clean.shape == raw.shape
    assert np.all(np.isfinite(clean))
    assert np.nanmean(clean[:, 0]) == pytest.approx(0.0, abs=1e-7)


def test_fit_characteristic_factor_model_smoke() -> None:
    characteristics = np.array(
        [
            [0.8, -0.2],
            [1.1, 0.4],
            [-0.6, 0.7],
            [0.2, -1.2],
        ]
    )
    returns = np.array([0.012, 0.018, -0.006, 0.002])

    model = fr.fit_characteristic_factor_model(returns, characteristics)

    assert len(model.factor_returns) == 2
    assert len(model.residual_variance) == 4
    assert all(v >= 0.0 for v in model.residual_variance)
