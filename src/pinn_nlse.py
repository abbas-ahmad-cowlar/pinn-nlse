"""
PINN-NLSE Model - Physics-Informed Neural Network for the NLSE
================================================================

Neural network that approximates the NLSE solution u(xi, tau) = a + i*b
by outputting (a, b) and computing the PDE residual via autograd.

Architecture:
    Input:  (xi, tau)        - 2 neurons
    Hidden: 5 layers x 128, tanh activation
    Output: (a, b)           - 2 neurons (linear, no activation)
