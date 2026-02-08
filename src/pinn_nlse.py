"""
PINN-NLSE Model - Physics-Informed Neural Network for the NLSE
================================================================

Neural network that approximates the NLSE solution u(xi, tau) = a + i*b
by outputting (a, b) and computing the PDE residual via autograd.

Architecture:
    Input:  (xi, tau)        - 2 neurons
    Hidden: 5 layers x 128, tanh activation
    Output: (a, b)           - 2 neurons (linear, no activation)

Sign Convention: Agrawal (s = -sign(beta_2) = +1 for anomalous)
    r_a = -d_xi b + (s/2) d_tau^2 a + N_sq * (a^2 + b^2) * a
    r_b = +d_xi a + (s/2) d_tau^2 b + N_sq * (a^2 + b^2) * b

"""

from __future__ import annotations

import torch
import torch.nn as nn


class PINN_NLSE(nn.Module):
    """Physics-Informed Neural Network for the Nonlinear Schrodinger Equation.

    Solves the normalized NLSE
        i * d_xi u + (s/2) * d_tau^2 u + N_sq * |u|^2 * u = 0
    by approximating u(xi, tau) = a(xi, tau) + i * b(xi, tau) with a feedforward MLP.

    Args:
        n_hidden: Number of hidden layers (default: 5)
        n_neurons: Neurons per hidden layer (default: 128)
        s: Sign parameter, -sign(beta_2) (default: 1, anomalous dispersion)
        N_sq: Soliton number squared, N^2 (default: 1.0)
        xi_max: Maximum xi in the training domain (used to scale inputs)
        tau_max: Maximum |tau| in the training domain (used to scale inputs)
    """

    def __init__(self, n_hidden: int = 5, n_neurons: int = 128,
                 s: float = 1, N_sq: float = 1.0,
                 xi_max: float = 5.0, tau_max: float = 20.0):
        super().__init__()

        self.s = s
        self.N_sq = N_sq
        self.xi_max = xi_max
        self.tau_max = tau_max

        layers: list[nn.Module] = []

        # Input layer: 2 -> n_neurons + tanh
        layers.append(nn.Linear(2, n_neurons))
        layers.append(nn.Tanh())

        # Hidden layers (n_hidden total tanh blocks; the input block above
        # already counts as the first one — add (n_hidden - 1) more).
        for _ in range(n_hidden - 1):
            layers.append(nn.Linear(n_neurons, n_neurons))
            layers.append(nn.Tanh())

        # Output layer: n_neurons -> 2 (no activation)
        layers.append(nn.Linear(n_neurons, 2))

        self.network = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self) -> None:
        """Apply Xavier/Glorot initialization to all linear layers."""
        for m in self.network:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def _scale_inputs(self, xi: torch.Tensor, tau: torch.Tensor):
        """Map normalized physical coordinates to approximately [-1, 1].

        Doing this inside the model decouples the network's internal scale
        from the physics ranges set in `config.py` (xi_max, tau_max).
        """
        xi_scaled = 2.0 * xi / self.xi_max - 1.0
        tau_scaled = tau / self.tau_max
        return xi_scaled, tau_scaled

    def forward(self, xi: torch.Tensor, tau: torch.Tensor):
        """Forward pass: predict (a, b) at given (xi, tau) coordinates.

        Args:
            xi: Tensor (N, 1), normalized distance in [0, xi_max]
            tau: Tensor (N, 1), normalized time in [-tau_max, +tau_max]

        Returns:
            a: Tensor (N, 1), real part of u
            b: Tensor (N, 1), imaginary part of u
        """
        xi_s, tau_s = self._scale_inputs(xi, tau)
        x = torch.cat([xi_s, tau_s], dim=1)  # (N, 2)
        out = self.network(x)                # (N, 2)
        a = out[:, 0:1]
        b = out[:, 1:2]
        return a, b

    def physics_residual(self, xi: torch.Tensor, tau: torch.Tensor):
        """Compute the NLSE physics residual using autograd.

        Computes:
            r_a = -d_xi b + (s/2) d_tau^2 a + N_sq * (a^2 + b^2) * a
            r_b = +d_xi a + (s/2) d_tau^2 b + N_sq * (a^2 + b^2) * b

        Args:
            xi: Tensor (N, 1) with requires_grad=True
            tau: Tensor (N, 1) with requires_grad=True

        Returns:
            r_a: Tensor (N, 1), real part of the residual
            r_b: Tensor (N, 1), imaginary part of the residual
        """
        a, b = self.forward(xi, tau)

        ones = torch.ones_like(a)

        # First derivatives in xi
        da_dxi = torch.autograd.grad(
            a, xi, grad_outputs=ones,
            create_graph=True, retain_graph=True,
        )[0]
        db_dxi = torch.autograd.grad(
            b, xi, grad_outputs=ones,
            create_graph=True, retain_graph=True,
        )[0]

        # Second derivatives in tau (chain: first derivative, then differentiate again)
        da_dtau = torch.autograd.grad(
            a, tau, grad_outputs=ones,
            create_graph=True, retain_graph=True,
        )[0]
        d2a_dtau2 = torch.autograd.grad(
            da_dtau, tau, grad_outputs=ones,
            create_graph=True, retain_graph=True,
        )[0]

        db_dtau = torch.autograd.grad(
            b, tau, grad_outputs=ones,
            create_graph=True, retain_graph=True,
        )[0]
        # Final derivative: omit retain_graph=True to reduce memory.
        d2b_dtau2 = torch.autograd.grad(
            db_dtau, tau, grad_outputs=ones,
            create_graph=True,
        )[0]

        intensity = a ** 2 + b ** 2  # |u|^2

        r_a = -db_dxi + (self.s / 2) * d2a_dtau2 + self.N_sq * intensity * a
        r_b = +da_dxi + (self.s / 2) * d2b_dtau2 + self.N_sq * intensity * b

        return r_a, r_b

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
