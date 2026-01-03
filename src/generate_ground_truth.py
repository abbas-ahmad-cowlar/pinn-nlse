"""
Ground-Truth Dataset Generator
================================

Runs the imported SSFM solver for the four PINN-NLSE test cases and writes
.npz datasets and physics-demo figures to data/ and figures/.

Cases (case_type):
  - soliton                  : N=1 fundamental soliton, sech IC, s=+1, N_sq=1
  - gaussian_dispersion      : Gaussian IC, s=-1, N_sq=0 (linear normal dispersion)
  - spm_no_dispersion        : Gaussian IC, s=0 (no-dispersion switch), N_sq=1
  - gaussian_nonlinear       : Gaussian IC, s=+1, N_sq=1 (optional, anomalous + Kerr)

Run from project root:
    python -m src.generate_ground_truth
or import generate_all() from notebooks.

"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapezoid

from src.config import (
    FIGURE_PATHS,
    N_SOLITON,
    N_T,
    N_Z,
    S_SIGN,
    TAU_MAX,
    XI_MAX,
)
from src.nlse_utils import compute_energy, create_grid, gaussian_pulse, sech_pulse
from src.ssfm import ssfm_propagate
from src.utils import plot_propagation_map, plot_spectrum_evolution

