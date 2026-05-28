from __future__ import annotations

import numpy as np
import pytest

import finance_risk as fr


def _returns() -> np.ndarray:
    return np.array(
        [
            [0.010, 0.004, -0.002],
            [0.012, 0.006, 0.001],
            [-0.006, -0.003, 0.004],
            [0.008, 0.002, -0.001],
            [-0.004, -0.005, 0.003],
            [0.015, 0.007, -0.002],
        ]
    )


def test_sample_covariance_keeps_assets_and_matches_numpy() -> None:
    estimate = fr.sample_covariance(_returns(), assets=["A", "B", "C"], frequency=1)

    assert estimate.method == "sample"
    assert estimate.assets == ("A", "B", "C")
    assert estimate.shrinkage is None
    assert estimate.matrix == pytest.approx(np.cov(_returns(), rowvar=False, ddof=1))


def test_shrinkage_covariances_are_psd_and_report_shrinkage() -> None:
    sample = fr.sample_covariance(_returns(), frequency=1).matrix
    ledoit = fr.ledoit_wolf_covariance(_returns(), frequency=1)
    oas = fr.oas_covariance(_returns(), frequency=1)

    for estimate in [ledoit, oas]:
        assert estimate.assets == ("asset_0", "asset_1", "asset_2")
        assert estimate.shrinkage is not None
        assert 0.0 <= estimate.shrinkage <= 1.0
        assert estimate.matrix == pytest.approx(estimate.matrix.T)
        assert np.linalg.eigvalsh(estimate.matrix).min() >= -1e-12

    assert abs(ledoit.matrix[0, 1]) <= abs(sample[0, 1])


def test_correlation_round_trip_and_nearest_psd() -> None:
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    corr = fr.covariance_to_correlation(cov)
    rebuilt = fr.correlation_to_covariance(corr, np.array([0.2, 0.3]))

    assert np.diag(corr).tolist() == pytest.approx([1.0, 1.0])
    assert rebuilt == pytest.approx(cov)

    not_psd = np.array([[1.0, 2.0], [2.0, 1.0]])
    fixed = fr.nearest_positive_semidefinite(not_psd)
    assert np.linalg.eigvalsh(fixed).min() >= -1e-12
