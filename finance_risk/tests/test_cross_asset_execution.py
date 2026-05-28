from __future__ import annotations

import numpy as np
import pytest

import finance_risk as fr


def test_cross_asset_correlation_cointegration_and_transfer_entropy() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    y = 2.0 * x + np.array([0.1, -0.1, 0.0, 0.1, -0.1, 0.0])

    rolling = fr.rolling_correlation(x, y, window=3)
    spread = fr.cointegration_spread(y, x)
    entropy = fr.transfer_entropy([0, 0, 1, 1, 0, 1, 1], [0, 1, 1, 0, 1, 1, 0], bins=2)

    assert np.isnan(rolling[:2]).all()
    assert rolling[-1] > 0.99
    assert spread.hedge_ratio == pytest.approx(2.0, rel=0.03)
    assert abs(spread.spread.mean()) < 1e-12
    assert entropy >= 0.0


def test_execution_models_return_complete_schedules() -> None:
    twap = fr.twap_schedule(total_quantity=100.0, periods=4)
    vwap = fr.vwap_schedule(total_quantity=100.0, volume_curve=[1.0, 2.0, 1.0])
    impact = fr.almgren_chriss_market_impact(quantity=50_000, volatility=0.02, daily_volume=1_000_000, spread=0.01)
    trajectory = fr.optimal_execution_trajectory(total_quantity=100.0, periods=5, risk_aversion=0.5)

    assert twap.tolist() == pytest.approx([25.0, 25.0, 25.0, 25.0])
    assert vwap.sum() == pytest.approx(100.0)
    assert vwap.tolist() == pytest.approx([25.0, 50.0, 25.0])
    assert impact.total_cost > impact.spread_cost > 0.0
    assert trajectory.remaining.tolist()[0] == pytest.approx(100.0)
    assert trajectory.remaining.tolist()[-1] == pytest.approx(0.0)
    assert trajectory.trades.sum() == pytest.approx(100.0)
