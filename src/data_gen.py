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
