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