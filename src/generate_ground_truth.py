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


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------

def git_commit_or_unknown(path: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", path, "rev-parse", "--short", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def ground_truth_metadata(u_hist, xi_arr, tau, omega, tau_max, dtau):
    if len(tau) > 1 and not np.isclose(float(tau[1] - tau[0]), float(dtau)):
        raise ValueError("dtau does not match the stored tau grid spacing")
    max_boundary_intensity = float(np.max(np.abs(u_hist[:, [0, -1]]) ** 2))
    return {
        "omega": omega,
        "dxi": float(xi_arr[1] - xi_arr[0]),
        "N_T": int(len(tau)),
        "N_Z": int(len(xi_arr) - 1),
        "tau_max": float(tau_max),
        "xi_max": float(xi_arr[-1]),
        "u_dtype": str(u_hist.dtype),
        "tau_dtype": str(tau.dtype),
        "max_boundary_intensity": max_boundary_intensity,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pinn_repo_commit": git_commit_or_unknown("."),
        "ssfm_source_commit": git_commit_or_unknown("../2-Split_Step_Fourier_Solver"),
        "ssfm_source_path": "../2-Split_Step_Fourier_Solver/src/nlse_ssfm",
    }


def assert_boundary_leakage_ok(u_hist, tolerance: float = 1e-6,
                               edge_points: int = 2, label: str = "dataset") -> float:
    edge_intensity = np.abs(np.concatenate(
        [u_hist[:, :edge_points], u_hist[:, -edge_points:]], axis=1
    )) ** 2
    max_edge = float(np.max(edge_intensity))
    if max_edge >= tolerance:
        raise AssertionError(
            f"{label}: boundary intensity {max_edge:.2e} exceeds {tolerance:.1e}; "
            "increase TAU_MAX or do not use zero temporal BCs for this case"
        )
    return max_edge


# ---------------------------------------------------------------------------
# Per-dataset generators