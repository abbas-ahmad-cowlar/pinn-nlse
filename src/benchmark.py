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
