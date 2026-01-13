"""
Tests for SSFM import wiring.

Verifies that the copied src/ssfm.py and src/nlse_utils.py expose the
companion-project API and produce correct soliton/energy/dispersion results.
"""

import inspect

import numpy as np
import pytest

from src.config import N_T, TAU_MAX, XI_MAX, S_SIGN
from src.nlse_utils import (
    compute_energy,
    create_grid,
    gaussian_pulse,
    sech_pulse,
)
from src.ssfm import dispersion_step, nonlinear_step, ssfm_propagate


def test_companion_sech_pulse_signature():
    sig = inspect.signature(sech_pulse)
    assert "amplitude" in sig.parameters, "sech_pulse must expose amplitude=1.0"
    assert "N" not in sig.parameters, (
        "Do not pass soliton order through sech_pulse; use ssfm_propagate(..., N_sq=N**2)"
    )

