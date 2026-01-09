"""
Tests for src.data_gen — collocation, IC, BC, and supervised data point generators.
"""

import numpy as np
import pytest
import torch

from src.config import N_BC_POINTS, N_COLLOCATION, N_IC_POINTS, TAU_MAX, XI_MAX
from src.data_gen import (
    assert_boundary_decay,
    assert_case_matches_model,
    generate_bc_points,
    generate_collocation_points,
    generate_data_points,
    generate_ic_points,
    load_ground_truth_npz,
)


# -- Collocation -----------------------------------------------------------------

def test_collocation_shape_and_grad():
    xi, tau = generate_collocation_points(N_COLLOCATION, XI_MAX, TAU_MAX, seed=0)
    assert xi.shape == (N_COLLOCATION, 1)
    assert tau.shape == (N_COLLOCATION, 1)
    assert xi.requires_grad is True
    assert tau.requires_grad is True
    assert xi.min() >= 0 and xi.max() <= XI_MAX
    assert tau.min() >= -TAU_MAX and tau.max() <= TAU_MAX


def test_collocation_seed_is_deterministic():
    xi1, tau1 = generate_collocation_points(256, XI_MAX, TAU_MAX, seed=42)
    xi2, tau2 = generate_collocation_points(256, XI_MAX, TAU_MAX, seed=42)
    assert torch.equal(xi1.detach(), xi2.detach())
    assert torch.equal(tau1.detach(), tau2.detach())


def test_collocation_sobol_method():
    xi, tau = generate_collocation_points(1024, XI_MAX, TAU_MAX, seed=7, method="sobol")
    assert xi.shape == (1024, 1)
    assert tau.shape == (1024, 1)
    assert xi.requires_grad is True
    assert tau.requires_grad is True

