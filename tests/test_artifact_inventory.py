"""
Published artifact inventory test.

Fails loudly if any required release artifact is missing — figure, notebook,
report, log, or model weights. Run as part of `pytest tests/`.
"""

from pathlib import Path

import pytest


REQUIRED_FIGURES = [
    "figures/published/gt_soliton_propagation.png",
    "figures/published/01_dispersion_broadening.png",
    "figures/published/02_spm_spectral_broadening.png",
    "figures/published/pinn_training_loss_soliton.png",
    "figures/published/pinn_training_loss_gaussian_dispersion.png",
    "figures/published/ssfm_convergence_study.png",
    "figures/published/ssfm_verification_soliton.png",
    "figures/comparison_soliton.png",
    "figures/error_map_soliton.png",
    "figures/cross_section_soliton.png",
    "figures/comparison_gaussian_dispersion.png",
    "figures/error_map_gaussian_dispersion.png",
    "figures/cross_section_gaussian_dispersion.png",
    "figures/speed_benchmark.png",
]


REQUIRED_TOP_LEVEL = [
    "notebooks/01_ssfm_validation.ipynb",
    "notebooks/02_pinn_training.ipynb",
    "notebooks/03_comparison.ipynb",
    "report/technical_report.md",
    "README.md",
    "logs/speed_benchmark.json",
    "logs/published/soliton_data_augmented_training_metadata.json",
    "logs/published/gaussian_dispersion_data_augmented_training_metadata.json",
    "data/provenance.json",
    "data/ssfm_validation_metrics.json",
]


@pytest.mark.parametrize("path", REQUIRED_FIGURES + REQUIRED_TOP_LEVEL)
def test_required_artifact_exists(path):
    """Every required deliverable must exist on disk."""
    assert Path(path).exists(), f"Missing required artifact: {path}"


def test_soliton_weights_present():
    """At least one soliton weight file must exist (data-augmented preferred)."""
    candidates = [
        Path("models/published/soliton_data_augmented_final.pt"),
        Path("models/published/soliton_final.pt"),
        Path("models/soliton_data_augmented_final.pt"),
        Path("models/soliton_final.pt"),
    ]
    assert any(p.exists() for p in candidates), (
        f"No soliton weights found among: {[str(p) for p in candidates]}"
    )


def test_gaussian_dispersion_weights_present():
    """At least one Gaussian dispersion weight file must exist."""
    candidates = [
        Path("models/published/gaussian_dispersion_data_augmented_final.pt"),
        Path("models/published/gaussian_dispersion_final.pt"),
        Path("models/gaussian_dispersion_data_augmented_final.pt"),
        Path("models/gaussian_dispersion_final.pt"),
    ]
    assert any(p.exists() for p in candidates), (
        f"No Gaussian dispersion weights found among: {[str(p) for p in candidates]}"
    )


def test_published_directories_have_readme_or_files():
    """Published/ directories must not be empty (artifact files or README)."""
    for d in ("models/published", "logs/published", "figures/published"):
        files = list(Path(d).iterdir()) if Path(d).exists() else []
        assert len(files) > 0, f"{d}/ is empty"