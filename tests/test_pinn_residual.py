"""
Tests for the PINN_NLSE physics residual.

Two analytical solutions of the normalized NLSE are constructed and fed to the
production PINN_NLSE.physics_residual method. If the residual signs are right,
both residual components must be ~0 (machine-precision-limited via autograd).
"""

import torch
import torch.nn as nn

from src.pinn_nlse import PINN_NLSE


class _AnalyticalSoliton(nn.Module):
    """Exact N=1 fundamental soliton: u(xi, tau) = sech(tau) * exp(i*xi/2)."""
    physics_residual = PINN_NLSE.physics_residual

    def __init__(self, s: float = 1, N_sq: float = 1.0):
        super().__init__()
        self.s = s
        self.N_sq = N_sq
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, xi, tau):
        sech_tau = 1.0 / torch.cosh(tau)
        a = sech_tau * torch.cos(xi / 2)
        b = sech_tau * torch.sin(xi / 2)
        return a, b


class _AnalyticalLinearGaussian(nn.Module):
    """Exact solution for i*u_xi + (s/2)*u_tautau = 0 with Gaussian IC."""
    physics_residual = PINN_NLSE.physics_residual

    def __init__(self, s: float = -1):
        super().__init__()
        self.s = s
        self.N_sq = 0.0
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, xi, tau):
        q = torch.complex(torch.ones_like(xi), self.s * xi)
        tau_c = torch.complex(tau, torch.zeros_like(tau))
        u = (1.0 / torch.sqrt(q)) * torch.exp(-(tau_c ** 2) / (2.0 * q))
        return torch.real(u), torch.imag(u)


def test_residual_zero_for_n1_soliton():
    torch.manual_seed(0)
    model = _AnalyticalSoliton(s=1, N_sq=1.0)
    xi = (5.0 * torch.rand(1000, 1)).requires_grad_(True)
    tau = ((2.0 * torch.rand(1000, 1) - 1.0) * 10.0).requires_grad_(True)
    r_a, r_b = model.physics_residual(xi, tau)
    assert torch.max(torch.abs(r_a)) < 1e-5
    assert torch.max(torch.abs(r_b)) < 1e-5


def test_residual_zero_for_linear_gaussian():
    torch.manual_seed(0)
    model = _AnalyticalLinearGaussian(s=-1)
    xi = (5.0 * torch.rand(1000, 1)).requires_grad_(True)
    tau = ((2.0 * torch.rand(1000, 1) - 1.0) * 10.0).requires_grad_(True)
    r_a, r_b = model.physics_residual(xi, tau)
    assert torch.max(torch.abs(r_a)) < 1e-5
    assert torch.max(torch.abs(r_b)) < 1e-5


def test_pinn_forward_and_param_count():
    torch.manual_seed(0)
    model = PINN_NLSE(n_hidden=5, n_neurons=128, s=1, N_sq=1.0,
                      xi_max=5.0, tau_max=20.0)
    assert model.count_parameters() == 66_690

    xi = torch.rand(64, 1, requires_grad=True)
    tau = torch.rand(64, 1, requires_grad=True)
    a, b = model(xi, tau)
    assert a.shape == (64, 1)
    assert b.shape == (64, 1)
    assert torch.isfinite(a).all() and torch.isfinite(b).all()


def test_pinn_residual_backprop():
    """Residual loss must produce non-zero gradients in the first linear layer."""
    torch.manual_seed(0)
    model = PINN_NLSE(n_hidden=5, n_neurons=128)
    xi = torch.rand(32, 1, requires_grad=True)
    tau = torch.rand(32, 1, requires_grad=True)
    r_a, r_b = model.physics_residual(xi, tau)
    loss = torch.mean(r_a ** 2 + r_b ** 2)
    loss.backward()
    grad = model.network[0].weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0
