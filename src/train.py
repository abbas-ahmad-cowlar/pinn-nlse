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


def _archive_existing(path: str) -> Optional[str]:
    """If path exists, rename it with a UTC timestamp suffix before overwriting.

    Provides a free safety net so that running training again does not silently
    wipe a previously published artifact. The archived file lives in the same
    directory and matches the gitignore pattern ``**/*.archived-*``, so it does
    not pollute the repository but is still recoverable on disk.
    """
    if not os.path.exists(path):
        return None
    stem, ext = os.path.splitext(path)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_path = f"{stem}.archived-{timestamp}{ext}"
    # os.replace is atomic on POSIX and Windows-safe (overwrites if target exists).
    os.replace(path, archive_path)
    print(f"  archived: {os.path.basename(path)} -> {os.path.basename(archive_path)}")
    return archive_path


def _safe_torch_save(state, path: str) -> None:
    """Archive any existing artifact at `path` before writing the new state."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    _archive_existing(path)
    torch.save(state, path)


def _safe_write_json(payload: dict, path: str) -> None:
    """Archive any existing artifact at `path` before writing the new JSON."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    _archive_existing(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _safe_save_figure(fig, path: str, dpi: int = 300) -> None:
    """Archive any existing figure at `path` before saving the new figure."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    _archive_existing(path)
    fig.savefig(path, dpi=dpi)


def save_checkpoint(model, optimizer, path: str, step: int, loss_dict: dict) -> None:
    """Save recoverable training state. Per-step checkpoints DO NOT auto-archive
    (they overwrite cheaply during a single run). Use :func:`_safe_torch_save`
    for canonical artifacts that should be preserved across runs.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "loss": loss_dict,
    }, path)


def make_lbfgs_data(data_dict: dict,
                    collocation_sampler: Optional[Callable] = None,
                    max_collocation: int = 5000) -> dict:
    """
    Return a memory-bounded full-batch dataset for L-BFGS refinement.

    If the Adam collocation set is larger than max_collocation, replace ONLY
    the collocation tensors with a fresh smaller Sobol/random set; keep IC,
    BC, and optional data labels unchanged.
    """
    out = dict(data_dict)
    n_current = int(out["xi_coll"].shape[0])
    if n_current > max_collocation:
        if collocation_sampler is None:
            raise ValueError(
                "collocation_sampler is required when reducing L-BFGS collocation size"
            )
        out["xi_coll"], out["tau_coll"] = collocation_sampler(max_collocation)
    return out


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------

def train_adam(model, data_dict, n_epochs, lr, lambdas, log_every=100,
               checkpoint_every=None, checkpoint_dir="models/checkpoints",
               case_name="model", collocation_sampler=None,
               resample_every=None, residual_val_data=None):
    """Train with Adam (stage 1: coarse convergence). Returns logged history."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[dict] = []

    for epoch in range(1, n_epochs + 1):
        if (collocation_sampler is not None and resample_every
                and epoch > 1 and epoch % resample_every == 0):
            data_dict["xi_coll"], data_dict["tau_coll"] = collocation_sampler()

        optimizer.zero_grad()
        clear_input_grads(data_dict)

        total_loss, loss_dict = compute_total_loss(
            model, **data_dict, lambdas=lambdas
        )

        if not torch.isfinite(total_loss):
            raise FloatingPointError(
                f"Non-finite Adam loss at epoch {epoch}: {loss_dict}"
            )

        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"Non-finite gradient norm at epoch {epoch}")
        optimizer.step()
        assert_finite_parameters(model, f"Adam epoch {epoch}")

        if checkpoint_every and epoch % checkpoint_every == 0:
            save_checkpoint(
                model, optimizer,
                os.path.join(checkpoint_dir, f"{case_name}_adam_{epoch:06d}.pt"),
                epoch, loss_dict
            )

        if epoch % log_every == 0 or epoch == 1:
            if residual_val_data is not None:
                r_a_val, r_b_val = model.physics_residual(
                    residual_val_data["xi_coll"], residual_val_data["tau_coll"]
                )
                loss_dict["phys_val"] = torch.mean(r_a_val ** 2 + r_b_val ** 2).item()
            history.append({"epoch": epoch, **loss_dict})
            print(f"[Adam {epoch:>6d}/{n_epochs}] "
                  f"Total: {loss_dict['total']:.6f} | "
                  f"Phys: {loss_dict['phys']:.6f} | "
                  f"IC: {loss_dict['ic']:.6f} | "
                  f"BC: {loss_dict['bc']:.6f}")

    return history

