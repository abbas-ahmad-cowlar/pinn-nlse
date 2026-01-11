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

