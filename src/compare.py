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
    xi_slices = [0.0, float(xi[-1]) / 2.0, float(xi[-1])]
    xi_indices = [int(np.argmin(np.abs(xi - x))) for x in xi_slices]
    fig_x, axes_x = plt.subplots(1, 3, figsize=(16, 4), sharey=True)
    for ax, idx, xi_val in zip(axes_x, xi_indices, xi_slices):
        ax.plot(tau_vis, int_ssfm[idx, tau_mask], "b-", lw=2.0, label="SSFM")
        ax.plot(tau_vis, int_pinn[idx, tau_mask], "r--", lw=2.0, label="PINN")
        ax.set_xlabel(r"$\tau$", fontsize=11)
        ax.set_title(rf"$\xi = {xi_val:.2f}$", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    axes_x[0].set_ylabel(r"$|u|^2$", fontsize=12)
    fig_x.suptitle(f"{title} - cross sections", fontsize=14)
    fig_x.tight_layout()
    xs_path = FIGURE_PATHS[cross_section_key]
    _safe_save_fig(fig_x, xs_path)
    plt.close(fig_x)

    return {"comparison": comp_path, "error_map": err_path, "cross_section": xs_path}


# ---------------------------------------------------------------------------
# Per-case orchestrators
# ---------------------------------------------------------------------------

def _ensure_ground_truth_present(data_path: str) -> None:
    """Auto-regenerate the SSFM ground-truth `.npz` files if any are missing.

    The ground-truth datasets are large (~10 MB each) and gitignored. On a fresh
    clone they will not exist; this helper transparently re-runs the
    deterministic generation pipeline so `python -m src.compare` and
    `notebooks/03_comparison.ipynb` work out-of-the-box without forcing the
    user to first open `notebooks/01_ssfm_validation.ipynb`. Generation is
    deterministic (seeded SSFM solves) and takes ~30 seconds on CPU.
    """
    if os.path.exists(data_path):
        return
    print(f"[compare] {data_path} missing — regenerating SSFM ground-truth datasets "
          f"via src.generate_ground_truth.generate_all() (~30 s on CPU)...")
    from src.generate_ground_truth import generate_all
    generate_all(include_optional=True)
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Auto-regeneration ran but {data_path} still missing — "
            "open notebooks/01_ssfm_validation.ipynb to investigate."
        )


def generate_case_comparison(case: str, data_path: str,
                             s: float, N_sq: float, ic_type: str,
                             comparison_key: str, error_key: str,
                             cross_section_key: str,
                             explicit_metadata_path: Optional[str] = None) -> dict:
    """Load one SSFM dataset + trained PINN, generate figures, return metrics dict."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _ensure_ground_truth_present(data_path)
    gt = load_ground_truth_npz(data_path)
    model = PINN_NLSE(
        n_hidden=5, n_neurons=128, s=s, N_sq=N_sq,
        xi_max=float(gt["xi_max"]), tau_max=float(gt["tau_max"]),
    ).to(device)
    assert_case_matches_model(gt, model, expected_ic_type=ic_type)

    model_path, metadata = resolve_model_path(case, explicit_metadata_path)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    u_pinn = evaluate_model_on_grid(model, gt["xi"], gt["tau"], device)
    u_ssfm = gt["u_hist"]

    aug_tag = " (data-augmented)" if "_data_augmented_" in model_path else ""
    title = f"{case}: PINN vs SSFM" + aug_tag
    paths = plot_case_figures(
        u_ssfm, u_pinn, gt["tau"], gt["xi"],
        comparison_key, error_key, cross_section_key, title,
    )

    metrics = compute_error_metrics(u_pinn, u_ssfm)
    pulse_mask = np.broadcast_to((np.abs(gt["tau"]) <= 10)[None, :], u_ssfm.shape)
    metrics["pulse_region_relative_l2"] = compute_masked_relative_l2_error(
        u_pinn, u_ssfm, pulse_mask,
    )
    # Cast NumPy floats to plain floats for clean printing/JSON export.
    metrics = {k: float(v) for k, v in metrics.items()}

    print(f"\n{case} metrics ({device})")
    print(f"  model_path        : {model_path}")
    print(f"  data_augmented    : {'_data_augmented_' in model_path}")
    if metadata:
        print(f"  metadata profile  : {metadata.get('training_profile', '?')}")
        print(f"  reported pulse-L2 : {metadata.get('pulse_region_relative_l2', '?')}")
    for k in ("relative_l2", "pulse_region_relative_l2", "mse", "max_pointwise", "mean_abs"):
        print(f"  {k:<18}: {metrics[k]:.6e}")

    return {
        "case": case,
        "model_path": model_path,
        "metadata": metadata,
        "metrics": metrics,
        "figure_paths": paths,
    }


def generate_soliton_comparison() -> dict:
    return generate_case_comparison(
        case="soliton",
        data_path="data/soliton_ground_truth.npz",
        s=1, N_sq=1.0, ic_type="sech",
        comparison_key="comparison_soliton",
        error_key="error_map_soliton",
        cross_section_key="cross_section_soliton",
    )


def generate_gaussian_dispersion_comparison() -> dict:
    return generate_case_comparison(
        case="gaussian_dispersion",
        data_path="data/dispersion_broadening_ground_truth.npz",
        s=-1, N_sq=0.0, ic_type="gaussian",
        comparison_key="comparison_gaussian_dispersion",
        error_key="error_map_gaussian_dispersion",
        cross_section_key="cross_section_gaussian_dispersion",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate PINN-NLSE comparison figures and metric tables.",
    )
    parser.add_argument(
        "--case",
        choices=["soliton", "gaussian_dispersion", "all"],
        default="soliton",
        help="Which case to render. 'all' regenerates both.",
    )
    args = parser.parse_args()

    if args.case in ("soliton", "all"):
        generate_soliton_comparison()
    if args.case in ("gaussian_dispersion", "all"):
        generate_gaussian_dispersion_comparison()


if __name__ == "__main__":
    main()
