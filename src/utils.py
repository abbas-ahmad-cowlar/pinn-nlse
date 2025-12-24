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


def plot_comparison(u_ssfm, u_pinn, tau, xi, xi_slice=None,
                    title="PINN vs SSFM", save_path=None,
                    show=True, tau_lim=None):
    """
    Side-by-side comparison of PINN prediction vs SSFM ground truth.

    Generates a 3-panel figure:
      Left:   |u_SSFM|^2 propagation map
      Center: |u_PINN|^2 propagation map
      Right:  log10|u_PINN - u_SSFM|^2 error map

    Args:
        u_ssfm: 2D complex array (N_z+1, N_t) - SSFM ground truth
        u_pinn: 2D complex array (N_z+1, N_t) - PINN prediction
        tau: 1D time array
        xi: 1D distance array
        xi_slice: Optional index for cross-section comparison overlay
        title: Figure title
        save_path: Optional save path
        show: Whether to call plt.show()
        tau_lim: Optional x-axis limits, e.g. (-10, 10) for the central pulse region.

    Returns:
        (fig, axes) matplotlib handles
    """
    import os
    import matplotlib.pyplot as plt

    int_ssfm = np.abs(u_ssfm) ** 2
    int_pinn = np.abs(u_pinn) ** 2
    error = np.abs(u_pinn - u_ssfm) ** 2

    if xi_slice is not None and not (0 <= xi_slice < len(xi)):
        raise IndexError(f"xi_slice={xi_slice} outside valid range [0, {len(xi)-1}]")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    im0 = axes[0].pcolormesh(tau, xi, int_ssfm, shading='auto', cmap='hot')
    axes[0].set_title("SSFM (Ground Truth)", fontsize=13)
    axes[0].set_xlabel(r'$\tau$'); axes[0].set_ylabel(r'$\xi$')
    if tau_lim is not None:
        axes[0].set_xlim(*tau_lim)
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].pcolormesh(tau, xi, int_pinn, shading='auto', cmap='hot')
    axes[1].set_title("PINN Prediction", fontsize=13)
    axes[1].set_xlabel(r'$\tau$')
    if tau_lim is not None:
        axes[1].set_xlim(*tau_lim)
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].pcolormesh(tau, xi, np.log10(error + 1e-16),
                             shading='auto', cmap='magma', vmin=-8, vmax=0)
    axes[2].set_title("Log Error: log10|PINN - SSFM|^2", fontsize=13)
    axes[2].set_xlabel(r'$\tau$')
    if tau_lim is not None:
        axes[2].set_xlim(*tau_lim)
    fig.colorbar(im2, ax=axes[2], label=r'$\log_{10}|u_{PINN}-u_{SSFM}|^2$')

    if xi_slice is not None:
        xi_val = xi[xi_slice]
        for ax in axes:
            ax.axhline(xi_val, color='white', lw=1.2, ls='--', alpha=0.8)
        inset = axes[2].inset_axes([0.08, -0.55, 0.84, 0.35])
        inset.plot(tau, int_ssfm[xi_slice], 'k-', lw=1.5, label='SSFM')
        inset.plot(tau, int_pinn[xi_slice], 'r--', lw=1.5, label='PINN')
        inset.set_title(fr'Cross-section at $\xi={xi_val:.2f}$', fontsize=10)
        inset.set_xlabel(r'$\tau$', fontsize=9)
        inset.set_ylabel(r'$|u|^2$', fontsize=9)
        inset.legend(fontsize=8)

    fig.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    return fig, axes


# ==============================================================
# Group 4: Error metrics
# ==============================================================

def compute_relative_l2_error(u_pred, u_true):
    """
    Compute the relative L2 error between predicted and true fields.

    Error = ||u_pred - u_true||_2 / ||u_true||_2

    Args:
        u_pred: Predicted field (complex array)
        u_true: True field (complex array)

    Returns:
        Relative L2 error (scalar, dimensionless)

    Raises:
        ValueError: if ||u_true|| = 0 (denominator would be zero)
    """
    denom = np.linalg.norm(u_true)
    if denom == 0:
        raise ValueError("Relative L2 denominator is zero.")
    return np.linalg.norm(u_pred - u_true) / denom


def compute_masked_relative_l2_error(u_pred, u_true, mask):
    """
    Compute relative L2 error only over a selected region.

    Use this for pulse-region metrics so low-amplitude window tails do not
    dominate the denominator or hide errors near the physical pulse.

    Args:
        u_pred: Predicted field (complex array)
        u_true: True field (complex array)
        mask: Boolean mask broadcastable to u_true / u_pred

    Returns:
        Relative L2 error on the selected mask
    """
    mask = np.broadcast_to(np.asarray(mask, dtype=bool), np.shape(u_true))
    u_pred_m = u_pred[mask]
    u_true_m = u_true[mask]
    denom = np.linalg.norm(u_true_m)
    if denom == 0:
        raise ValueError("Masked relative L2 denominator is zero.")
    return np.linalg.norm(u_pred_m - u_true_m) / denom


def compute_error_metrics(u_pred, u_true):
    """
    Compute comprehensive error metrics for PINN vs SSFM comparison.

    Args:
        u_pred: Predicted field (complex array)
        u_true: True field (complex array)

    Returns:
        dict with keys: 'mse', 'relative_l2', 'max_pointwise', 'mean_abs'
    """
    diff = u_pred - u_true
    return {
        'mse': np.mean(np.abs(diff) ** 2),
        'relative_l2': compute_relative_l2_error(u_pred, u_true),
        'max_pointwise': np.max(np.abs(diff)),
        'mean_abs': np.mean(np.abs(diff)),
    }