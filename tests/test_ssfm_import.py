"""
Tests for SSFM import wiring.

Verifies that the copied src/ssfm.py and src/nlse_utils.py expose the
companion-project API and produce correct soliton/energy/dispersion results.
"""

import inspect

import numpy as np
import pytest

from src.config import N_T, TAU_MAX, XI_MAX, S_SIGN
from src.nlse_utils import (
    compute_energy,
    create_grid,
    gaussian_pulse,
    sech_pulse,
)
from src.ssfm import dispersion_step, nonlinear_step, ssfm_propagate


def test_companion_sech_pulse_signature():
    sig = inspect.signature(sech_pulse)
    assert "amplitude" in sig.parameters, "sech_pulse must expose amplitude=1.0"
    assert "N" not in sig.parameters, (
        "Do not pass soliton order through sech_pulse; use ssfm_propagate(..., N_sq=N**2)"
    )


def test_create_grid_returns_tau_omega_dtau():
    tau, omega, dtau = create_grid(N_t=128, tau_window=10.0)
    assert tau.shape == (128,)
    assert omega.shape == (128,)
    assert tau[0] == pytest.approx(-10.0)
    assert tau[-1] < 10.0  # endpoint excluded
    assert dtau == pytest.approx(tau[1] - tau[0])
    expected_omega = 2 * np.pi * np.fft.fftfreq(128, d=dtau)
    np.testing.assert_allclose(omega, expected_omega)


def test_sech_pulse_energy_is_two():
    tau, _, dtau = create_grid(N_t=N_T, tau_window=TAU_MAX)
    u0 = sech_pulse(tau)
    assert compute_energy(u0, dtau) == pytest.approx(2.0, rel=1e-3)


def test_dispersion_and_nonlinear_steps_are_unitary():
    """Dispersion (frequency-domain phase) and Kerr (time-domain phase) preserve |u|^2."""
    tau, omega, dtau = create_grid(N_t=256, tau_window=15.0)
    u0 = sech_pulse(tau)
    E0 = compute_energy(u0, dtau)
    u_disp = dispersion_step(u0, omega, s=1, dxi_half=0.01)
    u_nl = nonlinear_step(u0, N_sq=1.0, dxi=0.01)
    assert compute_energy(u_disp, dtau) == pytest.approx(E0, rel=1e-12)
    assert compute_energy(u_nl, dtau) == pytest.approx(E0, rel=1e-12)


def test_ssfm_soliton_acid_test_high_resolution():
    """At N_z=2000 the pointwise intensity error should fall below 1e-6."""
    tau, omega, dtau = create_grid(N_t=N_T, tau_window=TAU_MAX)
    u0 = sech_pulse(tau)
    xi_arr, u_hist = ssfm_propagate(
        u0, tau, omega, xi_max=XI_MAX, N_z=2000, s=S_SIGN, N_sq=1.0,
    )
    u_exact = sech_pulse(tau)[None, :] * np.exp(0.5j * xi_arr[:, None])
    complex_err = float(np.max(np.abs(u_hist - u_exact)))
    intensity_err = float(np.max(np.abs(np.abs(u_hist) ** 2 - np.abs(u_exact) ** 2)))
    assert complex_err < 1e-3
    assert intensity_err < 1e-6


def test_ssfm_energy_conservation():
    tau, omega, dtau = create_grid(N_t=N_T, tau_window=TAU_MAX)
    u0 = sech_pulse(tau)
    xi_arr, u_hist = ssfm_propagate(
        u0, tau, omega, xi_max=XI_MAX, N_z=1000, s=S_SIGN, N_sq=1.0,
    )
    E0 = compute_energy(u_hist[0], dtau)
    drifts = [abs(compute_energy(u_hist[i], dtau) / E0 - 1.0) for i in range(len(xi_arr))]
    assert max(drifts) < 1e-10


def test_gaussian_dispersion_against_analytical():
    """Linear normal-dispersion case has a closed-form analytical solution."""
    tau, omega, dtau = create_grid(N_t=N_T, tau_window=TAU_MAX)
    u0 = gaussian_pulse(tau)
    xi_arr, u_hist = ssfm_propagate(
        u0, tau, omega, xi_max=XI_MAX, N_z=1000, s=-1, N_sq=0.0,
    )
    q = 1.0 + 1j * (-1) * xi_arr[:, None]
    u_exact = (1.0 / np.sqrt(q)) * np.exp(-(tau[None, :] ** 2) / (2.0 * q))
    rel_l2 = np.linalg.norm(u_hist[-1] - u_exact[-1]) / np.linalg.norm(u_exact[-1])
    assert rel_l2 < 1e-2
