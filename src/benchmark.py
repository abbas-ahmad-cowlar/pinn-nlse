"""
Speed Benchmark — SSFM vs PINN
================================

Honest fixed-case timing comparison between the SSFM solver and the trained
PINN. This is **not** a parameter-sweep benchmark — that would require a
parameter-conditioned PINN. We compare repeated SSFM solves of the
same physical case against repeated PINN inference on the same evaluation grid.

Three timings per N_runs are reported (median of `BENCHMARK_REPEATS` repeats):

- ``ssfm``: total wall time to call ``ssfm_propagate`` ``n_runs`` times
- ``pinn_forward``: total wall time for ``model(xi_grid, tau_grid)`` ``n_runs`` times
  (tensor grids constructed once, outside the timing loop)
- ``pinn_end_to_end``: total wall time including grid construction, host->device
  transfer, and host download per call. This is what an inverse-problem or
  real-time-control caller would see.

Modes:
- ``cpu_fair`` (default): SSFM and PINN both forced to CPU. Apples-to-apples.
- ``available_hardware``: PINN may use CUDA. Labeled explicitly in the figure.

"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Optional

import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.compare import resolve_model_path
from src.config import FIGURE_PATHS
from src.data_gen import assert_case_matches_model, load_ground_truth_npz
from src.nlse_utils import sech_pulse
from src.pinn_nlse import PINN_NLSE
from src.ssfm import ssfm_propagate


N_RUNS_LIST_DEFAULT = (1, 5, 10, 25, 50, 100)
BENCHMARK_REPEATS_DEFAULT = 5  # CPU-friendly; bump to 7-10 for the published number


def _summarize(samples: list[float]) -> dict:
    arr = np.asarray(samples, dtype=float)
    return {
        "median": float(np.median(arr)),
        "q1": float(np.percentile(arr, 25)),
        "q3": float(np.percentile(arr, 75)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
    }


def _archive_existing(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    stem, ext = os.path.splitext(path)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    new_path = f"{stem}.archived-{ts}{ext}"
    os.replace(path, new_path)
    return new_path


def run_speed_benchmark(
    benchmark_mode: str = "cpu_fair",
    n_runs_list: tuple[int, ...] = N_RUNS_LIST_DEFAULT,
    benchmark_repeats: int = BENCHMARK_REPEATS_DEFAULT,
    case: str = "soliton",
    figure_path: Optional[str] = None,
    log_path: str = "logs/speed_benchmark.json",
) -> dict:
    """Run the full speed benchmark and write figure + JSON. Returns the JSON dict."""
    assert benchmark_mode in ("cpu_fair", "available_hardware")
    assert case == "soliton", "Only soliton case is supported (the PINN is fixed-parameter)."

    if benchmark_mode == "cpu_fair":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    benchmark_label = f"{benchmark_mode}, PINN device={device.type}"

    # Load the SSFM grid we will benchmark on (same one used for the saved comparison)
    gt = load_ground_truth_npz("data/soliton_ground_truth.npz")
    tau = gt["tau"]
    omega = gt["omega"]
    xi_np = gt["xi"]
    xi_max = float(gt["xi_max"])
    n_z = int(gt["N_Z"])
    n_tau = len(tau)
    N_eval = len(xi_np)
    s_val = float(gt["s"])
    nsq = float(gt["N_sq"])
    u0 = sech_pulse(tau)

    # Load PINN (prefer published)
    model = PINN_NLSE(
        n_hidden=5, n_neurons=128, s=s_val, N_sq=nsq,
        xi_max=xi_max, tau_max=float(gt["tau_max"]),
    ).to(device)
    model_path, metadata = resolve_model_path(case)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    assert_case_matches_model(gt, model, expected_ic_type="sech")

    # Pre-construct the evaluation grid (shared across forward-only timings)
    xi_grid = torch.tensor(np.repeat(xi_np, n_tau), dtype=torch.float32).unsqueeze(1).to(device)
    tau_grid = torch.tensor(np.tile(tau, N_eval), dtype=torch.float32).unsqueeze(1).to(device)

    # Warm up
    with torch.no_grad():
        _ = model(xi_grid, tau_grid)
    if device.type == "cuda":
        torch.cuda.synchronize()

    print(f"\n=== Speed benchmark ({benchmark_label}) ===")
    print(f"  case            : {case}")
    print(f"  model_path      : {model_path}")
    print(f"  N_xi x N_tau    : {N_eval} x {n_tau}")
    print(f"  N_z (SSFM)      : {n_z}")
    print(f"  repeats per pt  : {benchmark_repeats}")
    print(f"  N_runs sweep    : {list(n_runs_list)}")
    print()

    ssfm_times, pinn_fwd_times, pinn_e2e_times = [], [], []
    ssfm_iqr, pinn_fwd_iqr, pinn_e2e_iqr = [], [], []
    records = []

    for n_runs in n_runs_list:
        ssfm_samples, pinn_fwd_samples, pinn_e2e_samples = [], [], []

        for _trial in range(benchmark_repeats):
            # SSFM: repeat the same physical solve N_runs times
            t0 = time.perf_counter()
            for _ in range(n_runs):
                _ = ssfm_propagate(u0, tau, omega, xi_max=xi_max, N_z=n_z, s=s_val, N_sq=nsq)
            ssfm_samples.append(time.perf_counter() - t0)

            # PINN forward-only (grid pre-built)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            with torch.no_grad():
                for _ in range(n_runs):
                    _, _ = model(xi_grid, tau_grid)
            if device.type == "cuda":
                torch.cuda.synchronize()
            pinn_fwd_samples.append(time.perf_counter() - t1)

            # PINN end-to-end: include tensor build + device transfer + host download
            if device.type == "cuda":
                torch.cuda.synchronize()
            t2 = time.perf_counter()
            with torch.no_grad():
                for _ in range(n_runs):
                    xg = torch.tensor(np.repeat(xi_np, n_tau), dtype=torch.float32).unsqueeze(1).to(device)
                    tg = torch.tensor(np.tile(tau, N_eval), dtype=torch.float32).unsqueeze(1).to(device)
                    a, b = model(xg, tg)
                    _ = a.cpu().numpy(), b.cpu().numpy()
            if device.type == "cuda":
                torch.cuda.synchronize()
            pinn_e2e_samples.append(time.perf_counter() - t2)

        s_summary = _summarize(ssfm_samples)
        f_summary = _summarize(pinn_fwd_samples)
        e_summary = _summarize(pinn_e2e_samples)

        ssfm_times.append(s_summary["median"])
        pinn_fwd_times.append(f_summary["median"])
        pinn_e2e_times.append(e_summary["median"])
        ssfm_iqr.append([s_summary["median"] - s_summary["q1"], s_summary["q3"] - s_summary["median"]])
        pinn_fwd_iqr.append([f_summary["median"] - f_summary["q1"], f_summary["q3"] - f_summary["median"]])
        pinn_e2e_iqr.append([e_summary["median"] - e_summary["q1"], e_summary["q3"] - e_summary["median"]])

        records.append({
            "n_runs": int(n_runs),
            "ssfm_samples_sec": [float(x) for x in ssfm_samples],
            "pinn_forward_samples_sec": [float(x) for x in pinn_fwd_samples],
            "pinn_end_to_end_samples_sec": [float(x) for x in pinn_e2e_samples],
            "ssfm_summary": s_summary,
            "pinn_forward_summary": f_summary,
            "pinn_end_to_end_summary": e_summary,
        })

        print(f"  N={n_runs:3d}  ssfm={s_summary['median']:.3f}s  "
              f"pinn_fwd={f_summary['median']:.3f}s  "
              f"pinn_e2e={e_summary['median']:.3f}s")

    # Plot: log-log with IQR error bars
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(n_runs_list, ssfm_times, yerr=np.array(ssfm_iqr).T,
                fmt="bo-", lw=2, ms=8, capsize=4, label="SSFM (median ± IQR)")
    ax.errorbar(n_runs_list, pinn_fwd_times, yerr=np.array(pinn_fwd_iqr).T,
                fmt="rs--", lw=2, ms=8, capsize=4, label="PINN forward-only")
    ax.errorbar(n_runs_list, pinn_e2e_times, yerr=np.array(pinn_e2e_iqr).T,
                fmt="m^-", lw=2, ms=8, capsize=4, label="PINN end-to-end")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of repeated fixed-case evaluations  $N_{\\rm runs}$", fontsize=13)
    ax.set_ylabel("Total computation time (s)", fontsize=13)
    ax.set_title(f"Speed benchmark: SSFM vs PINN ({benchmark_label})", fontsize=14)
    ax.legend(fontsize=11, loc="best")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    if figure_path is None:
        figure_path = FIGURE_PATHS["speed_benchmark"]
    os.makedirs(os.path.dirname(figure_path) or ".", exist_ok=True)
    _archive_existing(figure_path)
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # JSON log
    payload = {
        "benchmark_mode": benchmark_mode,
        "benchmark_label": benchmark_label,
        "device": str(device),
        "case": case,
        "model_path": model_path,
        "model_metadata_profile": metadata.get("training_profile"),
        "model_metadata_data_augmented": metadata.get("data_augmented"),
        "dataset_path": "data/soliton_ground_truth.npz",
        "grid": {
            "N_tau": int(n_tau),
            "N_xi": int(N_eval),
            "N_z": int(n_z),
            "xi_max": float(xi_max),
            "s": s_val,
            "N_sq": nsq,
        },
        "n_runs_list": list(map(int, n_runs_list)),
        "benchmark_repeats": int(benchmark_repeats),
        "ssfm_median_times": ssfm_times,
        "pinn_forward_median_times": pinn_fwd_times,
        "pinn_end_to_end_median_times": pinn_e2e_times,
        "records": records,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": (
            "Training time excluded; timing measures repeated fixed-case "
            "inference/solve after artifacts already exist. This is NOT a "
            "parameter-sweep benchmark — that would require a "
            "parameter-conditioned PINN."
        ),
    }
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    _archive_existing(log_path)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # Print headline table
    print(f"\n{'N_runs':>8} | {'SSFM (s)':>10} | {'PINN fwd (s)':>14} | {'PINN e2e (s)':>14} | {'fwd speedup':>12} | {'e2e speedup':>12}")
    print("-" * 80)
    for n, t_s, t_pf, t_pe in zip(n_runs_list, ssfm_times, pinn_fwd_times, pinn_e2e_times):
        fwd_sx = t_s / t_pf if t_pf > 0 else float("inf")
        e2e_sx = t_s / t_pe if t_pe > 0 else float("inf")
        print(f"{n:>8d} | {t_s:>10.3f} | {t_pf:>14.3f} | {t_pe:>14.3f} | {fwd_sx:>11.2f}x | {e2e_sx:>11.2f}x")

    print(f"\nSaved figure: {figure_path}")
    print(f"Saved log:    {log_path}")
    return payload


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SSFM vs PINN speed benchmark")
    parser.add_argument("--mode", choices=["cpu_fair", "available_hardware"], default="cpu_fair")
    parser.add_argument("--repeats", type=int, default=BENCHMARK_REPEATS_DEFAULT)
    args = parser.parse_args()
    run_speed_benchmark(benchmark_mode=args.mode, benchmark_repeats=args.repeats)