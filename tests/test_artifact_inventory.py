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