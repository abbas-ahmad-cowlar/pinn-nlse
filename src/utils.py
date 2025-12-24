"""
Utility Module - Plotting & Normalization
==========================================

Grid creation, sech(x), plotting helpers, error metrics.
Used by SSFM, PINN, and all notebooks.

Compatibility wrapper APIs (`create_time_grid`, `create_freq_grid`, `create_z_grid`)
are preserved on top of the canonical companion-project helpers in
`nlse_utils.py` so existing notebooks/tests keep working without changes.
"""

import numpy as np

from .nlse_utils import (
    create_grid,
    sech_pulse,
    gaussian_pulse,
    compute_energy,
    compute_spectrum,
)


# ==============================================================
# Group 1: Grid creation (compatibility wrappers around nlse_utils)
# ==============================================================

def create_time_grid(N_t, tau_max):
    """
    Create the normalized time grid for SSFM and PINN.

    Thin wrapper around `nlse_utils.create_grid` that returns only
    `(tau, dtau)`, preserving the compatibility API used by existing notebooks
    and tests. The grid spans [-tau_max, +tau_max) with N_t points
    (FFT-friendly endpoint exclusion).

    Args:
        N_t: Number of time grid points (must be a power of 2 per nlse_utils)
        tau_max: Half-width of time window

    Returns:
        tau: 1D array of normalized time values, shape (N_t,)
        dtau: Time step size (scalar)
    """
    tau, _omega, dtau = create_grid(N_t=N_t, tau_window=tau_max)
    return tau, dtau


def create_freq_grid(N_t, dtau):
    """
    Create the angular frequency grid for FFT-based dispersion.

    Returns angular frequencies in native FFT order (DC at index 0).

    Args:
        N_t: Number of time grid points
        dtau: Time step size

    Returns:
        omega: 1D array of angular frequencies, shape (N_t,)
    """
    return 2 * np.pi * np.fft.fftfreq(N_t, d=dtau)


def create_z_grid(N_z, xi_max):
    """
    Create the normalized propagation distance grid.

    Args:
        N_z: Number of propagation steps
        xi_max: Total propagation distance in units of L_D

    Returns:
        xi: 1D array of normalized distances, shape (N_z + 1,) including both endpoints [0, xi_max]
        dxi: Step size (scalar)
    """
    xi = np.linspace(0.0, xi_max, N_z + 1)
    dxi = xi[1] - xi[0]
    return xi, dxi


# ==============================================================
# Group 2: Pulse definitions
# ==============================================================

def sech(x):
    """
    Hyperbolic secant: sech(x) = 1/cosh(x), evaluated in an overflow-safe form.

    The fundamental soliton initial condition is u_0(tau) = sech(tau).
    Energy: integral of sech^2(tau) dtau = 2.0 (over all tau).

    Args:
        x: Input array or scalar

    Returns:
        sech(x): Same shape as input
    """
    x_abs = np.abs(x)
    exp_neg = np.exp(-x_abs)
    return 2.0 * exp_neg / (1.0 + exp_neg ** 2)


# ==============================================================
# Group 3: Plotting helpers
# ==============================================================

def plot_propagation_map(u_zt, tau, xi, title="Pulse Propagation",
                         save_path=None, show=True, tau_lim=None):
    """
    Plot a 2D colormap of |u(xi, tau)|^2 showing pulse evolution.

    Args:
        u_zt: 2D complex array of shape (N_z+1, N_t) - the field
        tau: 1D time array, shape (N_t,)
        xi: 1D distance array, shape (N_z+1,)
        title: Plot title string
        save_path: If provided, save figure to this path (300 dpi)
        show: Whether to call plt.show()
        tau_lim: Optional x-axis limits, e.g. (-10, 10) for the central pulse region.
                 If None, show the full available time window.

    Returns:
        (fig, ax) matplotlib handles
    """
    import os
    import matplotlib.pyplot as plt

    intensity = np.abs(u_zt) ** 2

    fig, ax = plt.subplots(figsize=(10, 6))