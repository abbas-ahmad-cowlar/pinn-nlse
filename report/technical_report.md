# PINN-NLSE: Physics-Informed Neural Networks for Nonlinear Fiber Optics

**Date**: 2026-05-10

## Abstract

We apply Physics-Informed Neural Networks (PINNs) to the normalized nonlinear
Schrödinger equation (NLSE), the master equation governing optical pulse
propagation in single-mode fibers, and benchmark them against a validated
split-step Fourier method (SSFM) ground-truth solver. Two regimes are
trained and reported: the N = 1 fundamental soliton (anomalous dispersion +
Kerr balance, exact analytical solution `u = sech(τ)·exp(iξ/2)`) and a pure
linear-dispersion Gaussian pulse (`s = −1`, `N² = 0`). The PINN outputs `(a, b)`
with `u = a + ib` and minimizes the NLSE residual via PyTorch autograd,
trained Adam → L-BFGS. Pure physics-only training drove the soliton case into
the trivial-solution attractor (`u → 0`); we recovered correctness with 500
SSFM supervision points (`λ_data = 1.0`) and explicitly label the result a
**data-augmented PINN**. Final pulse-region relative L2 error vs SSFM:
**1.29 % (soliton)** and **9.29 % (Gaussian)**. A speed benchmark on CPU shows
that for this 1D problem the SSFM remains ≈ 8× faster than the PINN at
fixed-case repeated inference; the PINN's value lies elsewhere (inverse
problems, parameter-conditioned extensions, continuous-coordinate evaluation,
differentiable physics).

## 1. Introduction

Pulse propagation in single-mode optical fibers is governed by the nonlinear
Schrödinger equation [1]:

$$
i\,\frac{\partial A}{\partial z}
\;=\; \frac{\beta_2}{2}\,\frac{\partial^2 A}{\partial \tau^2}
\;-\; \gamma\,|A|^2\,A,
$$

balancing group-velocity dispersion (`β₂`) against the Kerr nonlinearity (`γ`).
Together they produce the rich phenomenology of fiber optics — solitons,
self-phase-modulation-induced spectral broadening, modulational instability,
and ultimately supercontinuum generation in photonic crystal fibers.

The classical numerical solver is the **split-step Fourier method (SSFM)**:
operator-splitting between the dispersive step (computed in Fourier space, where
`∂²/∂τ² → −ω²` is multiplication) and the nonlinear step (a phase rotation in
the time domain). With Strang symmetric splitting it is second-order accurate
in the step size, conserves energy to machine precision, and produces the
N = 1 soliton to within `< 10⁻⁶` intensity error on a `1024 × 1000` grid.

A **Physics-Informed Neural Network (PINN)** [2] takes a fundamentally
different approach: instead of marching forward in `z`, it represents the
entire solution `u(ξ, τ)` as a single differentiable function (a neural
network), and tunes the network's weights until the function satisfies the
PDE residual at random sampled collocation points. Once trained, the network
produces a continuous, differentiable surrogate that can be queried at any
`(ξ, τ)`. This makes PINNs natural candidates for inverse problems and
parameter-conditioned forward problems [3, 4], even though they are typically
slower and less accurate than dedicated numerical solvers for a single
forward run.

This project applies the PINN methodology to the NLSE on two test cases — the
fundamental soliton and a Gaussian dispersion-only pulse — and compares the