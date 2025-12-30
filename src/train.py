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


def train_lbfgs(model, data_dict, n_steps, lambdas, log_every=100,
                checkpoint_every=50, checkpoint_dir="models/checkpoints",
                case_name="model", history_size=25):
    """Train with L-BFGS (stage 2: fine convergence). Bounded outer calls."""
    optimizer = torch.optim.LBFGS(
        model.parameters(), lr=1.0,
        max_iter=1, max_eval=5,
        tolerance_grad=1e-7, tolerance_change=1e-9,
        history_size=history_size, line_search_fn="strong_wolfe",
    )

    history: list[dict] = []

    for step in range(1, n_steps + 1):
        def closure():
            optimizer.zero_grad()
            clear_input_grads(data_dict)
            total_loss, loss_dict = compute_total_loss(
                model, **data_dict, lambdas=lambdas
            )
            if not torch.isfinite(total_loss):
                raise FloatingPointError(
                    f"Non-finite L-BFGS loss at outer step {step}: {loss_dict}"
                )
            total_loss.backward()
            return total_loss

        optimizer.step(closure)
        assert_finite_parameters(model, f"L-BFGS outer step {step}")

        if step % log_every == 0 or step == 1:
            # Re-evaluate AFTER the step (autograd, NOT no_grad — physics needs gradients)
            total_loss, loss_dict = compute_total_loss(
                model, **data_dict, lambdas=lambdas
            )
            del total_loss
            history.append({"epoch": step, **loss_dict})
            print(f"[LBFGS {step:>6d}/{n_steps}] "
                  f"Total: {loss_dict['total']:.6f} | "
                  f"Phys: {loss_dict['phys']:.6f} | "
                  f"IC: {loss_dict['ic']:.6f}")

        if checkpoint_every and step % checkpoint_every == 0:
            clear_input_grads(data_dict)
            total_loss, loss_dict = compute_total_loss(
                model, **data_dict, lambdas=lambdas
            )
            del total_loss
            save_checkpoint(
                model, optimizer,
                os.path.join(checkpoint_dir, f"{case_name}_lbfgs_{step:04d}.pt"),
                step, loss_dict
            )

    return history


# ---------------------------------------------------------------------------
# Profile resolution and shared helpers for case wrappers
# ---------------------------------------------------------------------------

def _resolve_training_profile(name: str) -> dict:
    """Convert a TRAINING_PROFILES entry from config.py to lowercase keys."""
    from src.config import TRAINING_PROFILES
    if name not in TRAINING_PROFILES:
        raise ValueError(
            f"Unknown training profile {name!r}; choose one of {sorted(TRAINING_PROFILES)}"
        )
    cfg = TRAINING_PROFILES[name]
    return {
        "n_collocation": cfg["N_COLLOCATION"],
        "n_ic": cfg["N_IC_POINTS"],
        "n_bc": cfg["N_BC_POINTS"],
        "adam_epochs": cfg["N_EPOCHS_ADAM"],
        "lbfgs_steps": cfg["N_STEPS_LBFGS"],
        "log_every": cfg["LOG_EVERY"],
        "lbfgs_history_size": cfg["LBFGS_HISTORY_SIZE"],
        "resample_every": cfg["RESAMPLE_EVERY"],
        "lbfgs_max_collocation": cfg["LBFGS_MAX_COLLOCATION"],
    }


def _assert_training_artifacts(outputs: dict) -> dict:
    """Fail loudly if a training wrapper returned before creating its artifacts."""
    for key in ("model_path", "history_path", "metadata_path"):
        path = outputs[key]
        if not os.path.exists(path):
            raise RuntimeError(f"Expected training artifact was not created: {path}")
    return outputs


def _smoke_preflight(seed: int, ic_func: str, s_value: int, N_sq: float,
                     xi_max: float, tau_max: float, lr: float, device,
                     lambdas: dict) -> None:
    """1k collocation / 500 Adam steps / no L-BFGS — verify finite, decreasing loss."""
    from src.pinn_nlse import PINN_NLSE
    from src.data_gen import (
        generate_collocation_points, generate_ic_points, generate_bc_points,
    )

    smoke_model = PINN_NLSE(n_hidden=3, n_neurons=64, s=s_value, N_sq=N_sq,
                            xi_max=xi_max, tau_max=tau_max).to(device)
    xi_sm, tau_sm = generate_collocation_points(1000, xi_max, tau_max, device,
                                                seed=seed * 100 + 1)
    xi_ic_sm, tau_ic_sm, a_ic_sm, b_ic_sm = generate_ic_points(
        100, tau_max, ic_func, device
    )
    xi_bc_sm, tau_bc_sm, a_bc_sm, b_bc_sm = generate_bc_points(
        50, xi_max, tau_max, device
    )
    smoke_data = {
        "xi_coll": xi_sm, "tau_coll": tau_sm,
        "xi_ic": xi_ic_sm, "tau_ic": tau_ic_sm, "a_ic": a_ic_sm, "b_ic": b_ic_sm,
        "xi_bc": xi_bc_sm, "tau_bc": tau_bc_sm, "a_bc": a_bc_sm, "b_bc": b_bc_sm,
    }
    hist_smoke = train_adam(smoke_model, smoke_data, 500, lr, lambdas, log_every=50)
    assert len(hist_smoke) >= 2, "Smoke run did not log enough loss values"
    smoke_totals = np.array([h["total"] for h in hist_smoke], dtype=float)
    if not np.all(np.isfinite(smoke_totals)):
        raise RuntimeError("Smoke run logged non-finite losses")
    window = max(2, len(smoke_totals) // 3)
    initial_median = float(np.median(smoke_totals[:window]))
    final_median = float(np.median(smoke_totals[-window:]))
    if final_median >= initial_median:
        raise RuntimeError(
            f"Smoke loss did not improve: initial median={initial_median:.3e}, "
            f"final median={final_median:.3e}"
        )
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _evaluate_on_grid(model, gt, device):
    """Predict (a, b) on the SSFM grid and return u_pinn (complex 2D array)."""
    model.eval()
    tau_np = gt["tau"]
    xi_np = gt["xi"]
    N_xi, N_tau = len(xi_np), len(tau_np)
    xi_grid = torch.tensor(np.repeat(xi_np, N_tau), dtype=torch.float32).unsqueeze(1).to(device)
    tau_grid = torch.tensor(np.tile(tau_np, N_xi), dtype=torch.float32).unsqueeze(1).to(device)
    with torch.no_grad():
        a_pred, b_pred = model(xi_grid, tau_grid)
    a_np = a_pred.cpu().numpy().reshape(N_xi, N_tau)
    b_np = b_pred.cpu().numpy().reshape(N_xi, N_tau)
    return a_np + 1j * b_np


# ---------------------------------------------------------------------------
# Case wrapper: N=1 soliton
# ---------------------------------------------------------------------------

def run_soliton_training(profile: str = "baseline", seed: int = 42,
                         skip_lbfgs: bool = False,
                         lbfgs_max_collocation: Optional[int] = None,
                         data_augmented: bool = False,
                         n_data_train: int = 500,
                         n_data_val: int = 1000,
                         run_tag: Optional[str] = None) -> dict:
    """Train the PINN on the N=1 soliton case.

    Args:
        data_augmented: If True, add SSFM supervision (lambda_data=1.0) with
            n_data_train points; metadata + artifacts are saved under the
            soliton_data_augmented_* names.
        n_data_train: Number of SSFM supervision points (used when data_augmented).
        n_data_val: Number of held-out SSFM points used for supervised
            validation MSE (always disjoint from training labels).
        run_tag: Optional run identifier appended to artifact filenames so
            independent verification runs do not overwrite the
            canonical published artifacts. Without ``run_tag``, existing
            canonical files are auto-archived with a UTC timestamp before
            being overwritten.

    Returns a dict with keys: case, model_path, history_path, metadata_path.
    Asserts the artifacts exist before returning.
    """
    import matplotlib
    if "ipykernel" not in sys.modules:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.pinn_nlse import PINN_NLSE
    from src.data_gen import (
        generate_collocation_points, generate_ic_points, generate_bc_points,
        generate_data_points,
        load_ground_truth_npz, assert_boundary_decay, assert_case_matches_model,
    )
    from src.config import (
        XI_MAX, TAU_MAX, S_SIGN, N_SOLITON, LEARNING_RATE,
        LAMBDA_PHYSICS, LAMBDA_IC, LAMBDA_BC, LAMBDA_DATA,
        CHECKPOINT_EVERY, FIGURE_PATHS,
    )
    from src.utils import compute_relative_l2_error, compute_masked_relative_l2_error

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[soliton] device = {device}{' (data-augmented)' if data_augmented else ''}")

    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    cfg = _resolve_training_profile(profile)
    lbfgs_max = (cfg["lbfgs_max_collocation"]
                 if lbfgs_max_collocation is None else lbfgs_max_collocation)

    def make_lambdas(data_weight=None):
        return {
            "phys": LAMBDA_PHYSICS, "ic": LAMBDA_IC, "bc": LAMBDA_BC,
            "data": LAMBDA_DATA if data_weight is None else data_weight,
        }

    smoke_lambdas = make_lambdas(data_weight=0.0)
    print(f"[soliton] running smoke preflight (1000 coll, 500 Adam steps)...")
    _smoke_preflight(seed, "sech", S_SIGN, float(N_SOLITON ** 2),
                     XI_MAX, TAU_MAX, LEARNING_RATE, device, smoke_lambdas)

    model = PINN_NLSE(n_hidden=5, n_neurons=128, s=S_SIGN,
                      N_sq=float(N_SOLITON ** 2),
                      xi_max=XI_MAX, tau_max=TAU_MAX).to(device)
    print(f"[soliton] params = {model.count_parameters():,}")

    gt = load_ground_truth_npz("data/soliton_ground_truth.npz")
    assert_boundary_decay(gt)
    assert_case_matches_model(gt, model, expected_ic_type="sech")

    coll_seed_state = {"value": seed}

    def collocation_sampler(n_points=None):
        coll_seed_state["value"] += 1
        n = cfg["n_collocation"] if n_points is None else n_points
        return generate_collocation_points(
            n, XI_MAX, TAU_MAX, device,
            seed=coll_seed_state["value"], method="sobol",
        )