"""
Comparison and Figure Generation CLI
=====================================

Regenerates the SSFM-vs-PINN comparison figures, error maps, cross-section
overlays, and quantitative metric tables for the soliton and Gaussian-
dispersion test cases. Also exposes the helper functions used by both the
CLI and ``notebooks/03_comparison.ipynb``.

Entry points:

    python -m src.compare --case soliton
    python -m src.compare --case gaussian_dispersion
    python -m src.compare --case all

This script is a thin wrapper around :func:`generate_case_comparison`. It does
NOT retrain anything — it loads the published / canonical PINN weights and
the saved SSFM ground-truth ``.npz`` files.

"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from typing import Optional

import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.config import FIGURE_PATHS
from src.data_gen import assert_case_matches_model, load_ground_truth_npz
from src.pinn_nlse import PINN_NLSE
from src.utils import compute_error_metrics, compute_masked_relative_l2_error


# ---------------------------------------------------------------------------
# Path resolution helpers (prefer published/, then data-augmented, then pure)
# ---------------------------------------------------------------------------

def _candidate_metadata_paths(case: str) -> list[str]:
    return [
        f"logs/published/{case}_data_augmented_training_metadata.json",
        f"logs/published/{case}_training_metadata.json",
        f"logs/{case}_data_augmented_training_metadata.json",
        f"logs/{case}_training_metadata.json",
    ]


def _candidate_model_paths(case: str) -> list[str]:
    return [
        f"models/published/{case}_data_augmented_final.pt",
        f"models/published/{case}_final.pt",
        f"models/{case}_data_augmented_final.pt",
        f"models/{case}_final.pt",
    ]


def resolve_model_path(case: str,
                       explicit_metadata_path: Optional[str] = None) -> tuple[str, dict]:
    """Pick a usable model path for ``case`` and return (path, metadata).

    Search order: explicit metadata file (if provided) -> published/_data_augmented
    -> published/_pure -> local _data_augmented -> local _pure. Returns the
    first existing checkpoint plus the matching metadata dict (or {} if no
    metadata is on disk).
    """
    metadata: dict = {}
    candidate_meta = (
        [explicit_metadata_path] if explicit_metadata_path else []
    ) + _candidate_metadata_paths(case)

    for p in candidate_meta:
        if p and os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            break

    # Always prefer the canonical search order (published/ first) so frozen
    # release artifacts are never silently bypassed in favor of local artifacts.
    # The metadata's `model_path` is a fallback for unusual layouts only.
    candidate_models = list(_candidate_model_paths(case))
    if metadata.get("model_path"):
        candidate_models.append(metadata["model_path"])

    for p in candidate_models:
        if p and os.path.exists(p):
            return p, metadata

    raise FileNotFoundError(
        f"No PINN weights found for case={case!r}. "
        f"Searched: {candidate_models}"
    )


# ---------------------------------------------------------------------------
# Forward pass on the SSFM grid
# ---------------------------------------------------------------------------

def evaluate_model_on_grid(model: PINN_NLSE,
                           xi: np.ndarray, tau: np.ndarray,
                           device: torch.device) -> np.ndarray:
    """Evaluate a trained PINN on the SSFM grid, returning a complex 2D array."""
    N_xi, N_tau = len(xi), len(tau)
    xi_grid = torch.tensor(np.repeat(xi, N_tau), dtype=torch.float32).unsqueeze(1).to(device)
    tau_grid = torch.tensor(np.tile(tau, N_xi), dtype=torch.float32).unsqueeze(1).to(device)
    model.eval()
    with torch.no_grad():
        a, b = model(xi_grid, tau_grid)
    a = a.cpu().numpy().reshape(N_xi, N_tau)
    b = b.cpu().numpy().reshape(N_xi, N_tau)
    return a + 1j * b


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def _archive_existing(path: str) -> Optional[str]:
    """Rename an existing file with a UTC timestamp before overwriting."""
    if not os.path.exists(path):
        return None
    stem, ext = os.path.splitext(path)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    new_path = f"{stem}.archived-{ts}{ext}"
    os.replace(path, new_path)
    return new_path


def _safe_save_fig(fig, path: str, dpi: int = 300) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    _archive_existing(path)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")


def plot_case_figures(u_ssfm: np.ndarray, u_pinn: np.ndarray,
                      tau: np.ndarray, xi: np.ndarray,
                      comparison_key: str, error_key: str,
                      cross_section_key: str, title: str,
                      tau_abs_max: float = 10.0) -> dict:
    """Generate the 3-panel comparison + error map + cross-section figures.

    Returns a dict with the paths actually written.
    """
    tau_mask = np.abs(tau) <= tau_abs_max
    tau_vis = tau[tau_mask]
    int_ssfm = np.abs(u_ssfm) ** 2
    int_pinn = np.abs(u_pinn) ** 2
    field_error = np.abs(u_pinn - u_ssfm) ** 2

    # 3-panel hero figure: SSFM | PINN | log10 error
    vmax = float(np.percentile(int_ssfm[:, tau_mask], 99))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    im0 = axes[0].pcolormesh(tau_vis, xi, int_ssfm[:, tau_mask],
                             cmap="inferno", shading="auto", vmin=0, vmax=vmax)
    axes[0].set_title("SSFM (Ground Truth)", fontsize=13)
    axes[0].set_xlabel(r"$\tau$", fontsize=12)
    axes[0].set_ylabel(r"$\xi = z/L_D$", fontsize=12)
    fig.colorbar(im0, ax=axes[0], label=r"$|u|^2$")

    im1 = axes[1].pcolormesh(tau_vis, xi, int_pinn[:, tau_mask],
                             cmap="inferno", shading="auto", vmin=0, vmax=vmax)
    axes[1].set_title("PINN Prediction", fontsize=13)
    axes[1].set_xlabel(r"$\tau$", fontsize=12)
    fig.colorbar(im1, ax=axes[1], label=r"$|u|^2$")

    im2 = axes[2].pcolormesh(
        tau_vis, xi, np.log10(field_error[:, tau_mask] + 1e-16),
        cmap="magma", shading="auto", vmin=-8, vmax=0,
    )
    axes[2].set_title(r"$\log_{10}|u_{\rm PINN} - u_{\rm SSFM}|^2$", fontsize=13)
    axes[2].set_xlabel(r"$\tau$", fontsize=12)
    fig.colorbar(im2, ax=axes[2], label=r"$\log_{10}\,$ error")
    fig.suptitle(title, fontsize=15)
    fig.tight_layout()
    comp_path = FIGURE_PATHS[comparison_key]
    _safe_save_fig(fig, comp_path)
    plt.close(fig)

    # Standalone error map
    fig_err, ax_err = plt.subplots(figsize=(10, 5))
    im_err = ax_err.pcolormesh(
        tau_vis, xi, np.log10(field_error[:, tau_mask] + 1e-16),
        cmap="magma", shading="auto", vmin=-8, vmax=0,
    )
    ax_err.set_xlabel(r"$\tau$", fontsize=12)
    ax_err.set_ylabel(r"$\xi = z/L_D$", fontsize=12)
    ax_err.set_title(f"{title} - error map", fontsize=14)
    fig_err.colorbar(im_err, ax=ax_err, label=r"$\log_{10}|u_{\rm PINN}-u_{\rm SSFM}|^2$")
    fig_err.tight_layout()
    err_path = FIGURE_PATHS[error_key]
    _safe_save_fig(fig_err, err_path)
    plt.close(fig_err)

    # Cross-section overlays at ξ = 0, ξ_max/2, ξ_max