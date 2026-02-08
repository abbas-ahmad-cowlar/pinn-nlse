"""
Tests for src.utils — Phase 1 wrappers around the companion nlse_utils helpers.
"""

import numpy as np
import pytest

from src.utils import (
    compute_error_metrics,
    compute_masked_relative_l2_error,
    compute_relative_l2_error,
    create_freq_grid,
    create_time_grid,
    create_z_grid,
    sech,
)


def test_create_time_grid_endpoint_excluded():
    tau, dtau = create_time_grid(1024, 20.0)
    assert len(tau) == 1024
    assert tau[0] == pytest.approx(-20.0)
    assert tau[-1] < 20.0  # endpoint=False
    assert dtau == pytest.approx(tau[1] - tau[0])


def test_create_freq_grid_dc_at_index_zero():
    omega = create_freq_grid(1024, 0.0390625)
    assert len(omega) == 1024
    assert omega[0] == pytest.approx(0.0)


def test_create_z_grid_includes_both_endpoints():
    xi, dxi = create_z_grid(1000, 5.0)
    assert len(xi) == 1001
    assert xi[0] == pytest.approx(0.0)
    assert xi[-1] == pytest.approx(5.0)
    assert dxi == pytest.approx(0.005)


def test_sech_zero_and_overflow_safety():
    assert sech(0.0) == pytest.approx(1.0)
    assert np.isfinite(sech(1000.0))
    tau, dtau = create_time_grid(1024, 20.0)
    energy = float(np.sum(sech(tau) ** 2) * dtau)
    assert energy == pytest.approx(2.0, rel=1e-2)


def test_relative_l2_error_one_percent_perturbation():
    tau, _ = create_time_grid(1024, 20.0)
    u_true = sech(tau).astype(complex)
    u_pred = u_true * 1.01
    err = compute_relative_l2_error(u_pred, u_true)
    assert err == pytest.approx(0.01, abs=1e-4)


def test_relative_l2_error_raises_on_zero_denominator():
    with pytest.raises(ValueError, match="denominator"):
        compute_relative_l2_error(np.ones(3), np.zeros(3))


def test_masked_relative_l2_error():
    tau, _ = create_time_grid(1024, 20.0)
    u_true = sech(tau).astype(complex)
    u_pred = u_true * 1.01
    mask = np.abs(tau) <= 10
    err = compute_masked_relative_l2_error(u_pred, u_true, mask)
    assert err == pytest.approx(0.01, abs=1e-4)


def test_compute_error_metrics_keys():
    u_true = np.ones(10, dtype=complex)
    u_pred = u_true * 1.01
    metrics = compute_error_metrics(u_pred, u_true)
    assert set(metrics.keys()) == {"mse", "relative_l2", "max_pointwise", "mean_abs"}
