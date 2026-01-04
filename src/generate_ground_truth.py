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
# ---------------------------------------------------------------------------

def _save_npz(path: str, **kwargs) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, **kwargs)
    print(f"  saved: {path}")


def _close_all_figs() -> None:
    plt.close("all")


def generate_soliton(tau, omega, dtau) -> tuple[np.ndarray, np.ndarray]:
    """N=1 fundamental soliton — primary PINN test case."""
    u0 = sech_pulse(tau)
    xi_arr, u_hist = ssfm_propagate(
        u0, tau, omega,
        xi_max=XI_MAX, N_z=N_Z,
        s=S_SIGN, N_sq=float(N_SOLITON ** 2),
    )
    assert_boundary_leakage_ok(u_hist, label="soliton")

    _save_npz(
        "data/soliton_ground_truth.npz",
        schema_version="1.1",
        case_name="soliton_N1",
        tau=tau, xi=xi_arr, u_hist=u_hist,
        dtau=dtau, s=S_SIGN, N_sq=float(N_SOLITON ** 2),
        ic_type="sech",
        case_type="soliton",
        dispersion_disabled=False,
        equation="i*u_xi + (s/2)*u_tautau + N_sq*|u|^2*u = 0",
        description="N=1 soliton propagation via SSFM (Agrawal convention, s=+1)",
        **ground_truth_metadata(u_hist, xi_arr, tau, omega, TAU_MAX, dtau),
    )

    plot_propagation_map(
        u_hist, tau, xi_arr,
        title=f"N={N_SOLITON} Soliton — SSFM Ground Truth",
        save_path=FIGURE_PATHS["gt_soliton_propagation"],
        show=False,
    )
    _close_all_figs()
    return xi_arr, u_hist


def generate_dispersion(tau, omega, dtau) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian pulse, normal dispersion, no nonlinearity. Validates against analytical."""
    u0 = gaussian_pulse(tau)
    xi_arr, u_hist = ssfm_propagate(
        u0, tau, omega,
        xi_max=XI_MAX, N_z=N_Z,
        s=-1, N_sq=0.0,
    )
    assert_boundary_leakage_ok(u_hist, label="dispersion-only Gaussian")

    # Analytical solution: u(xi,tau) = (1/sqrt(1 + i*s*xi)) * exp(-tau^2 / (2*(1 + i*s*xi)))
    q = 1.0 + 1j * (-1) * xi_arr[:, None]  # s = -1
    u_exact = (1.0 / np.sqrt(q)) * np.exp(-(tau[None, :] ** 2) / (2.0 * q))
    rel_l2 = (
        np.linalg.norm(u_hist[-1] - u_exact[-1]) /
        np.linalg.norm(u_exact[-1])
    )
    assert rel_l2 < 1e-2, f"dispersion analytical check failed: rel L2 = {rel_l2:.2e}"

    def rms_width(tau_arr, u):
        intensity = np.abs(u) ** 2
        norm = trapezoid(intensity, tau_arr, axis=-1)
        mean_tau = trapezoid(intensity * tau_arr[None, :], tau_arr, axis=-1) / norm
        variance = trapezoid(
            intensity * (tau_arr[None, :] - mean_tau[:, None]) ** 2,
            tau_arr, axis=-1,
        ) / norm
        return np.sqrt(variance)

    width_num = rms_width(tau, u_hist)
    width_exact = rms_width(tau, u_exact)
    width_rel_err = float(np.max(np.abs(width_num - width_exact) / width_exact))
    assert width_rel_err < 2e-2, f"dispersion width check failed: max rel err = {width_rel_err:.2e}"

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xi_arr, width_num, lw=2, label="SSFM")
    ax.plot(xi_arr, width_exact, "--", lw=1.5, label="Analytical")
    ax.set_xlabel(r"Normalized distance $\xi$")
    ax.set_ylabel("RMS pulse width")
    ax.set_title("Gaussian Dispersion: RMS Width vs Distance")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig.savefig(FIGURE_PATHS["dispersion_width"], dpi=300)
    print(f"  saved: {FIGURE_PATHS['dispersion_width']}")
    plt.close(fig)

    _save_npz(
        "data/dispersion_broadening_ground_truth.npz",
        schema_version="1.1",
        case_name="gaussian_dispersion_only",
        tau=tau, xi=xi_arr, u_hist=u_hist,
        dtau=dtau, s=-1, N_sq=0.0,
        ic_type="gaussian",
        case_type="gaussian_dispersion",
        dispersion_disabled=False,
        equation="i*u_xi + (s/2)*u_tautau + N_sq*|u|^2*u = 0",
        description="Gaussian pulse broadening with gamma=0 / N_sq=0 (s=-1, normal dispersion)",
        **ground_truth_metadata(u_hist, xi_arr, tau, omega, TAU_MAX, dtau),
    )

    plot_propagation_map(
        u_hist, tau, xi_arr,
        title="Gaussian Pulse Broadening - Dispersion Only",
        save_path=FIGURE_PATHS["dispersion_broadening"],
        show=False,
    )
    _close_all_figs()
    return xi_arr, u_hist


def generate_spm(tau, omega, dtau) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian pulse with pure Kerr nonlinearity (s=0 disables dispersion operator)."""
    u0 = gaussian_pulse(tau)
    xi_arr, u_hist = ssfm_propagate(
        u0, tau, omega,
        xi_max=5.0, N_z=N_Z,
        s=0, N_sq=1.0,  # computational switch: no dispersion operator
    )
    assert_boundary_leakage_ok(u_hist, label="SPM")

    # SPM must preserve temporal intensity exactly (it only rotates the phase)
    temporal_drift = float(np.max(np.abs(np.abs(u_hist[-1]) ** 2 - np.abs(u_hist[0]) ** 2)))
    assert temporal_drift < 1e-10, f"SPM changed temporal intensity: {temporal_drift:.2e}"

    spec = np.abs(np.fft.fftshift(np.fft.fft(u_hist[-1]))) ** 2
    edge_bins = max(4, len(spec) // 32)
    edge_fraction = float(
        (np.sum(spec[:edge_bins]) + np.sum(spec[-edge_bins:])) / np.sum(spec)
    )
    assert edge_fraction < 1e-4, (
        f"SPM spectrum reaches FFT boundary (edge fraction={edge_fraction:.2e}); "
        "increase TAU_MAX or N_T"
    )

    _save_npz(
        "data/spm_ground_truth.npz",
        schema_version="1.1",
        case_name="gaussian_spm_only",
        tau=tau, xi=xi_arr, u_hist=u_hist,
        dtau=dtau, s=0, N_sq=1.0,
        ic_type="gaussian",
        case_type="spm_no_dispersion",
        dispersion_disabled=True,
        equation="i*u_xi + (s/2)*u_tautau + N_sq*|u|^2*u = 0",
        description="Pure self-phase modulation with beta2=0; s=0 is a no-dispersion computational switch",
        **ground_truth_metadata(u_hist, xi_arr, tau, omega, TAU_MAX, dtau),
    )

    plot_spectrum_evolution(
        u_hist, tau, xi_arr,
        title="Self-Phase Modulation - Spectral Broadening",
        save_path=FIGURE_PATHS["spm_spectral_broadening"],
        normalize=True, log_scale=True, show=False,
    )
    _close_all_figs()
    return xi_arr, u_hist


def generate_nonlinear_gaussian(tau, omega, dtau,
                                tau_max_override: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Optional: Gaussian pulse, anomalous dispersion + Kerr (harder PINN test).

    Under (s=+1, N_sq=1) the Gaussian both broadens (dispersion) and
    self-modulates (Kerr), which can push intensity outside the default
    TAU_MAX=20 window. If `tau_max_override` is provided, the function
    rebuilds an extended grid for this dataset; otherwise it uses the same
    grid as the other cases and will fail boundary leakage check if leaky.
    """
    if tau_max_override is not None and tau_max_override != TAU_MAX:
        tau_local, omega_local, dtau_local = create_grid(
            N_t=N_T, tau_window=tau_max_override
        )
        tau_max_used = tau_max_override
    else:
        tau_local, omega_local, dtau_local = tau, omega, dtau
        tau_max_used = TAU_MAX

    u0 = gaussian_pulse(tau_local)
    xi_arr, u_hist = ssfm_propagate(
        u0, tau_local, omega_local,
        xi_max=XI_MAX, N_z=N_Z,
        s=S_SIGN, N_sq=float(N_SOLITON ** 2),
    )
    assert_boundary_leakage_ok(u_hist, label="nonlinear Gaussian")
    # Rebind names so the rest of the function uses the (possibly extended) grid.
    tau, omega, dtau = tau_local, omega_local, dtau_local
    TAU_MAX_LOCAL = tau_max_used
    # We rebind the module-level TAU_MAX only inside the metadata call:
    # the metadata helper takes tau_max as an explicit argument.

    _save_npz(
        "data/gaussian_nonlinear_ground_truth.npz",
        schema_version="1.1",
        case_name="gaussian_nonlinear_anomalous",
        tau=tau, xi=xi_arr, u_hist=u_hist,
        dtau=dtau, s=S_SIGN, N_sq=float(N_SOLITON ** 2),
        ic_type="gaussian",
        case_type="gaussian_nonlinear",
        dispersion_disabled=False,
        equation="i*u_xi + (s/2)*u_tautau + N_sq*|u|^2*u = 0",
        description="Gaussian pulse in anomalous dispersion with N_sq=1 via SSFM",
        **ground_truth_metadata(u_hist, xi_arr, tau, omega, TAU_MAX_LOCAL, dtau),
    )

    plot_propagation_map(
        u_hist, tau, xi_arr,
        title="Gaussian Pulse in Anomalous Dispersion - SSFM Ground Truth",
        save_path=FIGURE_PATHS["gt_gaussian_nonlinear"],
        show=False,
    )
    _close_all_figs()
    return xi_arr, u_hist


# ---------------------------------------------------------------------------
# Verification of saved data
# ---------------------------------------------------------------------------

REQUIRED_KEYS = [
    "schema_version", "case_name", "tau", "xi", "omega", "u_hist",
    "dtau", "dxi", "N_T", "N_Z", "tau_max", "xi_max",
    "s", "N_sq", "ic_type", "case_type", "dispersion_disabled", "equation",
    "u_dtype", "tau_dtype", "max_boundary_intensity", "generated_at_utc",
    "pinn_repo_commit", "ssfm_source_commit", "ssfm_source_path",
]


def verify_saved_data(paths: list[str]) -> None:
    for path in paths:
        data = np.load(path, allow_pickle=False)
        for key in REQUIRED_KEYS:
            assert key in data, f"{path} missing required key: {key}"
        assert data["tau"].shape == (N_T,), f"{path}: tau shape mismatch"
        assert data["omega"].shape == (N_T,), f"{path}: omega shape mismatch"
        assert data["u_hist"].shape == (len(data["xi"]), N_T), f"{path}: u_hist shape mismatch"
        assert int(data["N_T"]) == N_T, f"{path}: N_T metadata mismatch"
        assert int(data["N_Z"]) == len(data["xi"]) - 1, f"{path}: N_Z metadata mismatch"
        assert np.isclose(float(data["dxi"]), data["xi"][1] - data["xi"][0]), f"{path}: dxi mismatch"
        assert np.iscomplexobj(data["u_hist"]), f"{path}: u_hist must be complex"
        assert float(data["max_boundary_intensity"]) < 1e-6, f"{path}: boundary leakage too large"

    # SPM-specific guards
    spm = np.load("data/spm_ground_truth.npz", allow_pickle=False)
    assert str(np.asarray(spm["case_type"])) == "spm_no_dispersion"
    assert bool(np.asarray(spm["dispersion_disabled"])) is True
    assert float(np.asarray(spm["s"])) == 0.0, "SPM uses s=0 only as a no-dispersion switch"

    # Soliton complex/intensity acid test (using config N_z; this is bulk dataset, not 2.2 verification)
    soliton = np.load("data/soliton_ground_truth.npz", allow_pickle=False)
    u_exact = (1.0 / np.cosh(soliton["tau"]))[None, :] * np.exp(0.5j * soliton["xi"][:, None])
    complex_err = float(np.max(np.abs(soliton["u_hist"] - u_exact)))
    intensity_err = float(np.max(np.abs(np.abs(soliton["u_hist"]) ** 2 - np.abs(u_exact) ** 2)))
    # N_z=1000 default: complex_err ~ 1e-5, intensity_err ~ 4e-6 (Strang O(dxi^2))
    assert complex_err < 1e-3, f"saved soliton complex error: {complex_err:.2e}"
    assert intensity_err < 1e-5, f"saved soliton intensity error: {intensity_err:.2e}"
    print(f"  soliton bulk: complex_err={complex_err:.2e}, intensity_err={intensity_err:.2e}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_all(include_optional: bool = True,
                 optional_tau_max: float = 30.0,
                 verify: bool = True) -> dict:
    """Generate all ground-truth datasets and return a summary dict.

    The optional Gaussian + anomalous + Kerr case broadens beyond the default
    TAU_MAX=20 window; we use a wider window (default 30) just for that dataset.
    """
    os.makedirs("data", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    tau, omega, dtau = create_grid(N_t=N_T, tau_window=TAU_MAX)
    print(f"Grid: N_T={N_T}, tau_window={TAU_MAX}, dtau={dtau:.6f}, N_z={N_Z}, xi_max={XI_MAX}")

    print("\n[1/4] N=1 soliton")
    generate_soliton(tau, omega, dtau)
    print("\n[2/4] Gaussian dispersion-only")
    generate_dispersion(tau, omega, dtau)
    print("\n[3/4] Self-phase modulation (s=0 switch)")
    generate_spm(tau, omega, dtau)

    saved = [
        "data/soliton_ground_truth.npz",
        "data/dispersion_broadening_ground_truth.npz",
        "data/spm_ground_truth.npz",
    ]

    if include_optional:
        print(f"\n[4/4] Optional: Gaussian + anomalous + Kerr (tau_window={optional_tau_max} for boundary safety)")
        try:
            generate_nonlinear_gaussian(tau, omega, dtau, tau_max_override=optional_tau_max)
            saved.append("data/gaussian_nonlinear_ground_truth.npz")
        except AssertionError as exc:
            print(f"  WARNING: optional dataset skipped due to boundary leakage: {exc}")

    if verify:
        print("\nVerifying saved data...")
        verify_saved_data(saved)

    summary = {
        "saved_paths": saved,
        "config": {
            "N_T": int(N_T), "N_Z": int(N_Z),
            "tau_max": float(TAU_MAX), "xi_max": float(XI_MAX),
            "s_default": int(S_SIGN), "N_soliton": int(N_SOLITON),
        },
        "verified": bool(verify),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open("data/ground_truth_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved: data/ground_truth_summary.json")
    return summary


if __name__ == "__main__":
    generate_all()