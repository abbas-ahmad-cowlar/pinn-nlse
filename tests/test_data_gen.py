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


def test_collocation_unknown_method_raises():
    with pytest.raises(ValueError, match="Unknown collocation"):
        generate_collocation_points(10, XI_MAX, TAU_MAX, method="halton")


# -- IC -------------------------------------------------------------------------

def test_ic_sech_peak_and_zero_imag():
    xi, tau, a, b = generate_ic_points(N_IC_POINTS, TAU_MAX, "sech")
    assert torch.all(xi == 0)
    assert torch.allclose(a[N_IC_POINTS // 2], torch.tensor([[1.0]]), atol=0.01)
    assert torch.all(b == 0)
    # endpoint exclusion: the last sampled tau must be < +tau_max
    assert tau.max() < TAU_MAX


def test_ic_gaussian_variant():
    xi, tau, a, b = generate_ic_points(100, TAU_MAX, "gaussian")
    assert torch.allclose(a[50], torch.tensor([[1.0]]), atol=0.01)
    assert torch.all(b == 0)


def test_ic_unknown_function_raises():
    with pytest.raises(ValueError, match="Unknown IC"):
        generate_ic_points(10, TAU_MAX, "lorentzian")


# -- BC -------------------------------------------------------------------------

def test_bc_split_and_zero_labels():
    xi, tau, a, b = generate_bc_points(N_BC_POINTS, XI_MAX, TAU_MAX)
    assert xi.shape == (N_BC_POINTS, 1)
    assert torch.all(a == 0) and torch.all(b == 0)
    assert torch.any(tau == -TAU_MAX) and torch.any(tau == TAU_MAX)


# -- Data sampler --------------------------------------------------------------

def test_load_ground_truth_npz_validates_schema():
    data = load_ground_truth_npz("data/soliton_ground_truth.npz")
    assert "u_hist" in data
    assert np.iscomplexobj(data["u_hist"])


def test_assert_boundary_decay_passes_on_soliton():
    data = load_ground_truth_npz("data/soliton_ground_truth.npz")
    # Should not raise; sech(20) tail amplitude squared is ~1.7e-17.
    assert_boundary_decay(data) < 1e-6


def test_data_points_are_disjoint_with_held_out_indices():
    data = load_ground_truth_npz("data/soliton_ground_truth.npz")
    xi_t, tau_t, a_t, b_t, train_idx = generate_data_points(
        data["u_hist"], data["xi"], data["tau"], N_data=400, seed=11, return_indices=True,
    )
    assert xi_t.shape == (400, 1)
    assert a_t.dtype == torch.float32

    xi_v, tau_v, a_v, b_v, val_idx = generate_data_points(
        data["u_hist"], data["xi"], data["tau"], N_data=200, seed=22,
        exclude_flat_indices=set(train_idx.tolist()),
        return_indices=True,