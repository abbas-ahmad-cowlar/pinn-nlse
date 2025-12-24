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
    im = ax.pcolormesh(tau, xi, intensity, shading='auto', cmap='hot')
    ax.set_xlabel(r'Normalized time $\tau$', fontsize=14)
    ax.set_ylabel(r'Normalized distance $\xi = z/L_D$', fontsize=14)
    ax.set_title(title, fontsize=16)
    if tau_lim is not None:
        ax.set_xlim(*tau_lim)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r'$|u(\xi, \tau)|^2$', fontsize=12)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    if show:
        plt.show()
    return fig, ax


def plot_spectrum_evolution(u_zt, tau, xi, title="Spectral Evolution",
                            save_path=None, show=True, omega_lim=None,
                            normalize=True, log_scale=False, floor_db=-80):
    """
    Plot spectral evolution |u_tilde(xi, omega)|^2 as a 2D colormap.

    Args:
        u_zt: 2D complex array (N_z+1, N_t)
        tau: 1D time array (N_t,)
        xi: 1D distance array (N_z+1,)
        title: Plot title
        save_path: Optional save path
        show: Whether to call plt.show()
        omega_lim: Optional x-axis limits for frequency. If None, show the full spectrum.
        normalize: If True, divide spectrum by its global maximum before plotting.
        log_scale: If True, plot 10*log10(normalized spectrum) with floor_db clipping.
        floor_db: Lower plotting floor for log_scale.

    Returns:
        (fig, ax) matplotlib handles
    """
    import os
    import matplotlib.pyplot as plt

    dtau = tau[1] - tau[0]
    omega = np.fft.fftshift(2 * np.pi * np.fft.fftfreq(len(tau), d=dtau))

    spec = np.zeros_like(np.abs(u_zt) ** 2)
    for i in range(u_zt.shape[0]):
        spec[i, :] = np.abs(dtau * np.fft.fftshift(np.fft.fft(u_zt[i, :]))) ** 2

    if normalize:
        spec = spec / max(np.max(spec), np.finfo(float).tiny)

    if log_scale:
        plot_spec = 10 * np.log10(np.maximum(spec, 10 ** (floor_db / 10)))
        cbar_label = r'$10\log_{10}(|\tilde{u}|^2 / \max|\tilde{u}|^2)$ [dB]'
        vmin, vmax = floor_db, 0
    else:
        plot_spec = spec
        cbar_label = (r'$|\tilde{u}(\xi, \omega)|^2$'
                      if not normalize else r'Normalized $|\tilde{u}(\xi, \omega)|^2$')
        vmin, vmax = None, None

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.pcolormesh(omega, xi, plot_spec, shading='auto', cmap='viridis',
                       vmin=vmin, vmax=vmax)
    ax.set_xlabel(r'Normalized frequency $\omega$', fontsize=14)
    ax.set_ylabel(r'$\xi = z/L_D$', fontsize=14)
    ax.set_title(title, fontsize=16)
    if omega_lim is not None:
        ax.set_xlim(*omega_lim)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label, fontsize=12)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    return fig, ax

