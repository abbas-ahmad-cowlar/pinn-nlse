"""
Smoke test for the PINN training loop.

Runs `compute_total_loss` + a few Adam steps on a tiny dataset to confirm
finite-loss + decreasing-loss + gradient flow. Should complete in seconds.
"""

import numpy as np
import torch

from src.config import S_SIGN, N_SOLITON, XI_MAX, TAU_MAX, LEARNING_RATE
from src.data_gen import (
    generate_collocation_points,
    generate_ic_points,
    generate_bc_points,
)
from src.pinn_nlse import PINN_NLSE
from src.train import compute_total_loss, train_adam


def _build_tiny_problem(seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")
    model = PINN_NLSE(n_hidden=3, n_neurons=64, s=S_SIGN,
                      N_sq=float(N_SOLITON ** 2),
                      xi_max=XI_MAX, tau_max=TAU_MAX).to(device)
    xi_c, tau_c = generate_collocation_points(256, XI_MAX, TAU_MAX, device, seed=1)
    xi_ic, tau_ic, a_ic, b_ic = generate_ic_points(64, TAU_MAX, "sech", device)
    xi_bc, tau_bc, a_bc, b_bc = generate_bc_points(32, XI_MAX, TAU_MAX, device)
    data_dict = {
        "xi_coll": xi_c, "tau_coll": tau_c,
        "xi_ic": xi_ic, "tau_ic": tau_ic, "a_ic": a_ic, "b_ic": b_ic,
        "xi_bc": xi_bc, "tau_bc": tau_bc, "a_bc": a_bc, "b_bc": b_bc,
    }
    lambdas = {"phys": 1.0, "ic": 10.0, "bc": 1.0, "data": 0.0}
    return model, data_dict, lambdas


def test_compute_total_loss_returns_finite_per_term():
    model, data_dict, lambdas = _build_tiny_problem()
    total, parts = compute_total_loss(model, **data_dict, lambdas=lambdas)
    assert torch.isfinite(total)
    for key in ("phys", "ic", "bc", "data", "total"):
        assert key in parts
        assert np.isfinite(parts[key])


def test_train_adam_decreases_total_loss():
    model, data_dict, lambdas = _build_tiny_problem()
    history = train_adam(model, data_dict, n_epochs=80,
                         lr=LEARNING_RATE, lambdas=lambdas, log_every=10)
    assert len(history) >= 4
    totals = np.array([h["total"] for h in history], dtype=float)
    assert np.all(np.isfinite(totals))
    window = max(2, len(totals) // 3)
    assert np.median(totals[-window:]) < np.median(totals[:window]), (
        f"loss did not decrease: head={totals[:window].tolist()}, "
        f"tail={totals[-window:].tolist()}"
    )


def test_data_loss_term_raises_when_unsupplied_with_weight():
    model, data_dict, lambdas = _build_tiny_problem()
    lambdas_d = dict(lambdas, data=1.0)
    try:
        compute_total_loss(model, **data_dict, lambdas=lambdas_d)
    except ValueError:
        return
    raise AssertionError("compute_total_loss must raise when lambda_data > 0 but data tensors are missing")
