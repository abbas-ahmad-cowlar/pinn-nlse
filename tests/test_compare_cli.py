"""
Tests for the comparison CLI / module.

Exercises `src.compare`'s helpers and CLI surface against the published
artifacts. Designed to be cheap (loads pretrained weights, no training) so
it stays inside `pytest tests/`'s budget.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_compare_imports_cleanly():
    """The module + key helpers must import without side effects."""
    from src.compare import (  # noqa: F401
        evaluate_model_on_grid,
        generate_case_comparison,
        generate_gaussian_dispersion_comparison,
        generate_soliton_comparison,
        plot_case_figures,
        resolve_model_path,
    )


def test_resolve_model_path_prefers_published():
    """When both published/ and local model files exist, published/ wins."""
    from src.compare import resolve_model_path

    if not Path("models/published/soliton_data_augmented_final.pt").exists():
        pytest.skip("no published soliton weights to test resolver against")

    path, _meta = resolve_model_path("soliton")
    assert "published" in str(path).replace("\\", "/"), (
        f"resolver should prefer models/published/ but returned: {path}"
    )


def test_compare_cli_help_works():
    """CLI must at least show its help text without crashing."""
    result = subprocess.run(
        [sys.executable, "-m", "src.compare", "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, f"compare --help failed: {result.stderr}"
    assert "--case" in result.stdout
    assert "soliton" in result.stdout
    assert "gaussian_dispersion" in result.stdout


@pytest.mark.timeout(180)
def test_soliton_comparison_runs_and_produces_metrics():
    """End-to-end: load published weights, evaluate, return metrics."""
    if not Path("models/published/soliton_data_augmented_final.pt").exists():
        pytest.skip("no published soliton weights")
    if not Path("data/soliton_ground_truth.npz").exists():
        pytest.skip("no soliton ground truth .npz")

    # Force a non-interactive backend before anything imports matplotlib.
    os.environ["MPLBACKEND"] = "Agg"
    from src.compare import generate_soliton_comparison

    result = generate_soliton_comparison()

    assert "metrics" in result
    m = result["metrics"]
    # Headline number from the report (1.29 % pulse-region rel L2)
    assert m["pulse_region_relative_l2"] < 0.05, (
        f"Pulse-region rel L2 unexpectedly high: {m['pulse_region_relative_l2']}"
    )
    assert m["relative_l2"] < 0.05
    assert m["mse"] < 1e-3
    # Expected output figures exist after the call
    for key in ("comparison", "error_map", "cross_section"):
        assert Path(result["figure_paths"][key]).exists()


@pytest.mark.timeout(180)
def test_gaussian_dispersion_comparison_runs():
    if not Path("models/published/gaussian_dispersion_data_augmented_final.pt").exists():
        pytest.skip("no published gaussian weights")
    if not Path("data/dispersion_broadening_ground_truth.npz").exists():
        pytest.skip("no gaussian dispersion ground truth .npz")

    os.environ["MPLBACKEND"] = "Agg"
    from src.compare import generate_gaussian_dispersion_comparison

    result = generate_gaussian_dispersion_comparison()
    m = result["metrics"]
    # Plan threshold: pulse-region rel L2 < 10 %.
    assert m["pulse_region_relative_l2"] < 0.10, (
        f"Pulse-region rel L2 above 10 %: {m['pulse_region_relative_l2']}"
    )