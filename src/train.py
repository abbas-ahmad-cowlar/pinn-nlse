"""
PINN-NLSE Training Script
==========================

Trains the PINN model on the NLSE using a two-stage optimizer strategy:
  Stage 1: Adam (coarse convergence)
  Stage 2: L-BFGS (fine convergence; bounded full-batch outer calls)

Loss = lambda_phys * L_physics + lambda_ic * L_IC + lambda_bc * L_BC + lambda_data * L_data

"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Callable, Optional

import numpy as np
import torch

# Imports moved into wrappers / main to keep module-level cost light:
#   from src.pinn_nlse import PINN_NLSE
#   from src.config import ...
#   from src.data_gen import ...
#   from src.utils import ...


# ---------------------------------------------------------------------------
# Loss + helpers
# ---------------------------------------------------------------------------

def compute_total_loss(model, xi_coll, tau_coll,
                       xi_ic, tau_ic, a_ic, b_ic,
                       xi_bc, tau_bc, a_bc, b_bc,
                       xi_data=None, tau_data=None,
                       a_data=None, b_data=None,
                       lambdas: Optional[dict] = None):
    """
    Compute the total PINN loss with all 4 components.

    Returns (total_loss_tensor, loss_dict) where loss_dict has the per-term
    floats so you can log which term is failing during training.
    """
    if lambdas is None:
        lambdas = {"phys": 1.0, "ic": 10.0, "bc": 1.0, "data": 0.0}

    # Physics loss
    r_a, r_b = model.physics_residual(xi_coll, tau_coll)
    L_phys = torch.mean(r_a ** 2 + r_b ** 2)

    # IC loss
    a_pred_ic, b_pred_ic = model(xi_ic, tau_ic)
    L_ic = torch.mean((a_pred_ic - a_ic) ** 2 + (b_pred_ic - b_ic) ** 2)

    # BC loss
    a_pred_bc, b_pred_bc = model(xi_bc, tau_bc)
    L_bc = torch.mean((a_pred_bc - a_bc) ** 2 + (b_pred_bc - b_bc) ** 2)

    # Data loss (optional)
    L_data = torch.tensor(0.0, device=xi_coll.device)
    if lambdas["data"] > 0:
        if any(v is None for v in (xi_data, tau_data, a_data, b_data)):
            raise ValueError(
                "lambda_data > 0 requires xi_data, tau_data, a_data, and b_data"
            )
        a_pred_d, b_pred_d = model(xi_data, tau_data)
        L_data = torch.mean((a_pred_d - a_data) ** 2 + (b_pred_d - b_data) ** 2)

    total = (lambdas["phys"] * L_phys +
             lambdas["ic"] * L_ic +
             lambdas["bc"] * L_bc +
             lambdas["data"] * L_data)

    if not torch.isfinite(total):
        raise FloatingPointError("Non-finite PINN loss encountered")

    loss_dict = {
        "total": total.item(),
        "phys": L_phys.item(),
        "ic": L_ic.item(),
        "bc": L_bc.item(),
        "data": L_data.item(),
    }
    return total, loss_dict


def clear_input_grads(data_dict: dict) -> None:
    """Clear gradients on coordinate tensors that require autograd derivatives."""
    for value in data_dict.values():
        if torch.is_tensor(value) and value.grad is not None:
            value.grad = None


def assert_finite_parameters(model, context: str) -> None:
    """Raise immediately if an optimizer step creates NaN/Inf weights."""
    for name, param in model.named_parameters():
        if not torch.isfinite(param).all():
            raise FloatingPointError(f"Non-finite parameter after {context}: {name}")

