"""
PINN Training Data Generators
==============================

Generates the four types of training points for the PINN-NLSE model:
1. Collocation points: random (xi, tau) for physics loss (no labels)
2. Initial condition points: (xi=0, tau) with labels from u_0
3. Boundary condition points: (xi, tau=+/-tau_max) with labels = 0
4. Data points (optional): (xi, tau) with labels from SSFM solution

All outputs are PyTorch tensors with requires_grad as needed.

"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Generator 1: Collocation points (physics loss, no labels)
# ---------------------------------------------------------------------------

def generate_collocation_points(N_coll: int, xi_max: float, tau_max: float,
                                device: str = "cpu",
                                seed: Optional[int] = None,
                                method: str = "random"):
    """
    Generate random collocation points for physics loss.

    These points have NO labels — the loss is the PDE residual computed from
    the PINN's own output via autograd. requires_grad=True so PyTorch can
    differentiate through xi and tau when computing the residual.

    Args:
        N_coll: Number of collocation points
        xi_max: Max propagation distance
        tau_max: Max time (half-width)
        device: PyTorch device ('cpu' or 'cuda')
        seed: Optional seed for reproducible sampling
        method: 'random' or 'sobol' low-discrepancy sampling

    Returns:
        xi_coll: Tensor (N_coll, 1), requires_grad=True
        tau_coll: Tensor (N_coll, 1), requires_grad=True
    """
    if method == "random":
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed))
        xi_unit = torch.rand(N_coll, 1, device=device, generator=generator)
        tau_unit = torch.rand(N_coll, 1, device=device, generator=generator)
    elif method == "sobol":
        # SobolEngine seeds via the seed argument; if None it auto-randomizes.
        engine = torch.quasirandom.SobolEngine(
            dimension=2, scramble=True,
            seed=int(seed) if seed is not None else None,
        )
        pts = engine.draw(N_coll).to(device)
        xi_unit = pts[:, 0:1]
        tau_unit = pts[:, 1:2]
    else:
        raise ValueError(f"Unknown collocation sampling method: {method}")

    xi = xi_unit * xi_max
    tau = (tau_unit * 2 - 1) * tau_max
    xi.requires_grad_(True)
    tau.requires_grad_(True)
    return xi, tau


# ---------------------------------------------------------------------------
# Generator 2: Initial condition points (xi = 0)
# ---------------------------------------------------------------------------

def generate_ic_points(N_ic: int, tau_max: float, ic_func: str = "sech",
                       device: str = "cpu"):
    """
    Generate initial condition points at xi = 0.

    Uses the SSFM/FFT-grid convention: includes -tau_max, excludes +tau_max.

    Args:
        N_ic: Number of IC points
        tau_max: Max time (half-width)
        ic_func: Initial condition type ('sech' or 'gaussian')
        device: PyTorch device

    Returns:
        xi_ic: Tensor (N_ic, 1), all zeros (xi = 0)
        tau_ic: Tensor (N_ic, 1), uniformly spaced on [-tau_max, +tau_max)
        a_ic: Tensor (N_ic, 1), real part of u_0
        b_ic: Tensor (N_ic, 1), imaginary part of u_0
    """
    xi = torch.zeros(N_ic, 1, device=device)
    # Match SSFM FFT grid convention: include -tau_max, exclude +tau_max.
    tau = torch.linspace(-tau_max, tau_max, N_ic + 1, device=device)[:-1].unsqueeze(1)

    if ic_func == "sech":
        a = 1.0 / torch.cosh(tau)
        b = torch.zeros_like(tau)
    elif ic_func == "gaussian":
        a = torch.exp(-tau ** 2 / 2)
        b = torch.zeros_like(tau)
    else:
        raise ValueError(f"Unknown IC function: {ic_func}")

    return xi, tau, a, b


# ---------------------------------------------------------------------------
# Generator 3: Boundary condition points (tau = +/- tau_max)
# ---------------------------------------------------------------------------

def generate_bc_points(N_bc: int, xi_max: float, tau_max: float,
                       device: str = "cpu"):
    """
    Generate boundary condition points at tau = +/- tau_max.

    Coordinate note: +tau_max is an analytical far-field point, not a stored
    SSFM FFT-grid point. The grid itself excludes +tau_max; this is checked
    separately in assert_boundary_decay() for any data we want to use as
    zero-BC training labels.

    The BC enforces u -> 0 at the temporal boundaries. Because sech(20) ~ 4e-9
    and Gaussian(20) ~ 1e-87, this is effectively a Dirichlet zero condition.

    Args:
        N_bc: Number of BC points (split equally between -tau_max and +tau_max)
        xi_max: Max propagation distance
        tau_max: Max time (half-width)
        device: PyTorch device

    Returns:
        xi_bc: Tensor (N_bc, 1), random xi values in [0, xi_max]
        tau_bc: Tensor (N_bc, 1), values at -tau_max or +tau_max
        a_bc: Tensor (N_bc, 1), all zeros (target)
        b_bc: Tensor (N_bc, 1), all zeros (target)
    """
    N_half = N_bc // 2
    xi = torch.rand(N_bc, 1, device=device) * xi_max

    tau_left = -tau_max * torch.ones(N_half, 1, device=device)
    tau_right = tau_max * torch.ones(N_bc - N_half, 1, device=device)
    tau = torch.cat([tau_left, tau_right], dim=0)

    a = torch.zeros(N_bc, 1, device=device)
    b = torch.zeros(N_bc, 1, device=device)

    return xi, tau, a, b