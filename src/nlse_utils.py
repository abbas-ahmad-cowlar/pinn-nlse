"""
NLSE Utility Module
====================

IMPORTED from companion project: ../2-Split_Step_Fourier_Solver/
This module was built and validated in the companion SSFM project.
It is copied here verbatim to make the PINN-NLSE project self-contained.

Shared functions for the Split-Step Fourier Solver project:
- Grid creation (time and frequency domains)
- Pulse definitions (Gaussian, sech/soliton)
- Energy computation
- Spectrum computation
- RMS width and instantaneous-frequency diagnostics
- Plotting helpers for propagation maps

The PINN project's `src/utils.py` re-exports the helpers needed downstream
(`create_grid`, `sech_pulse`, `gaussian_pulse`, `compute_energy`,
`compute_spectrum`) and keeps compatibility wrapper APIs intact.

Source: Split-Step Fourier Solver companion project
Source commit: see data/provenance.json -> ssfm_source_commit
"""

import numpy as np

