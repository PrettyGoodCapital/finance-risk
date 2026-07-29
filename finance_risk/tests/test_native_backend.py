from __future__ import annotations

import numpy as np
import pytest

from finance_risk import finance_risk as native


def test_native_backend_exposes_risk_kernels() -> None:
    sample = native.sample_covariance_matrix([[0.01, 0.02], [0.02, 0.04], [0.03, 0.06]], 1.0)
    twap = native.twap_schedule(100.0, 4)

    np.testing.assert_allclose(sample, [[0.0001, 0.0002], [0.0002, 0.0004]])
    assert twap == pytest.approx([25.0, 25.0, 25.0, 25.0])


def test_public_covariance_delegates_to_native_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    from finance_risk import covariance

    called: dict[str, object] = {}

    def fake_sample(data: list[list[float]], frequency: float) -> list[list[float]]:
        called["data"] = data
        called["frequency"] = frequency
        return [[0.04, 0.0], [0.0, 0.09]]

    monkeypatch.setattr(covariance._native, "sample_covariance_matrix", fake_sample)

    estimate = covariance.sample_covariance(np.ones((3, 2)), assets=["A", "B"], frequency=12)

    assert called == {"data": [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]], "frequency": 12.0}
    assert estimate.assets == ("A", "B")
    np.testing.assert_allclose(estimate.matrix, [[0.04, 0.0], [0.0, 0.09]])
