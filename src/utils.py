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