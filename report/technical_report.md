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
results against the SSFM ground truth honestly. The repository includes the
code, notebooks, frozen published artifacts, tests, and reproduction
instructions needed to inspect the result as a standalone scientific Python
project.

## 2. Methods

### 2.1 The normalized NLSE

We work entirely in normalized units (`τ = t/T₀`, `ξ = z/L_D`, `u = A/√P₀`),
in which the NLSE collapses to:

$$
i\,\frac{\partial u}{\partial \xi}
\;+\; \frac{s}{2}\,\frac{\partial^2 u}{\partial \tau^2}
\;+\; N^2\,|u|^2\,u
\;=\; 0,
\qquad s = -\,\mathrm{sign}(\beta_2),
\qquad N^2 = \frac{L_D}{L_{\rm NL}}.
$$

Anomalous dispersion (`β₂ < 0`, `s = +1`) plus `N = 1` produces the fundamental
soliton with the exact closed-form solution `u(ξ, τ) = sech(τ)·exp(iξ/2)`. We
adopt this convention throughout — the same one used by Agrawal [1] and the
companion SSFM project — and verify the residual implementation against this
analytical answer before training (see §2.4).

### 2.2 Split-step Fourier ground truth

The SSFM is imported from a separate companion project (`2-Split_Step_Fourier_Solver/`)
and validated on the PINN repository's environment. Its dispersion substep is
`exp(−i s ω² h / 2)` (the minus sign comes from `∂²/∂τ² → −ω²`); the nonlinear
substep is `u·exp(+i N² |u|² h)`. With Strang splitting `D/2 → N → D/2`, the
single-step error is `O(dξ³)` and the global error is `O(dξ²)`.

On a `N_t = 1024`, `tau_window = 20`, `xi_max = 5`, `N_z = 2000` grid, the
imported SSFM reproduces the analytical soliton to:

- max complex error `|u_SSFM − sech(τ)·exp(iξ/2)|` = **2.8 × 10⁻⁶**
- max intensity error `||u_SSFM|² − sech²(τ)|` = **9.7 × 10⁻⁷**
- energy conservation drift `max |E(ξ)/E(0) − 1|` = **6.3 × 10⁻¹³**
- Strang convergence slope (fitted, log–log): **2.00**

These metrics are saved in `data/ssfm_validation_metrics.json` and visualized
in `figures/published/ssfm_convergence_study.png`.

### 2.3 PINN architecture

The complex field is split into real-valued outputs `u = a + ib`. Substituting
into the normalized NLSE and separating real and imaginary parts gives the
**two real residuals** the network must minimize:

$$
r_a = -\,\partial_\xi b + \tfrac{s}{2}\,\partial^2_\tau a + N^2(a^2 + b^2)\,a,
\qquad
r_b = +\,\partial_\xi a + \tfrac{s}{2}\,\partial^2_\tau b + N^2(a^2 + b^2)\,b.
$$

The negative sign on `∂_ξ b` in `r_a` comes from `i² = −1` and is the most
common implementation error in PINN-NLSE codes; it is double-checked against
the analytical residual test in §2.4.

The network architecture is a fully-connected MLP with input `(ξ, τ)`, five
hidden layers of 128 neurons each with `tanh` activation, and a 2-output linear
head — **66 690 trainable parameters** exactly. `tanh` is mandatory: ReLU has a
zero second derivative almost everywhere, which would silently delete the
dispersion term from the residual. Inputs are scaled to `[−1, +1]` inside the
network so the physical ranges (`ξ_max = 5`, `τ_max = 20`) do not push the tanh
units into saturation. Xavier/Glorot initialization is applied to all linear
layers.

### 2.4 Loss function and training

The total loss is a weighted sum of four terms, evaluated on three (optionally
four) sets of training points:

$$
\mathcal{L}
= \lambda_{\rm phys}\,\overline{r_a^2 + r_b^2}
+ \lambda_{\rm ic}\,\mathrm{MSE}_{\xi=0}
+ \lambda_{\rm bc}\,\mathrm{MSE}_{\tau=\pm \tau_{\max}}
+ \lambda_{\rm data}\,\mathrm{MSE}_{\rm SSFM}.
$$

Default weights: `λ_phys = 1`, `λ_ic = 10`, `λ_bc = 1`, `λ_data = 0` (pure PINN).

- **Collocation points** (5 000 by default, requires gradients): random `(ξ, τ)`
  in `[0, 5] × [−20, +20]` sampled with a scrambled Sobol sequence. The physics
  residual is evaluated here.
- **IC points** (250 by default): `ξ = 0`, with labels `a₀(τ) = sech(τ)` (or
  `exp(−τ²/2)` for the Gaussian case) and `b₀ = 0`.
- **BC points** (100 by default): `τ = ±20`, with zero Dirichlet labels. Valid
  because the IC pulse decays to `~ 4 × 10⁻⁹` (sech case) at the boundary; the
  `assert_boundary_decay` check enforces this.
- **Data points** (optional 500): random selections from the SSFM ground-truth
  grid restricted to the pulse region (`|τ| ≤ 10`, `|u|² ≥ 10⁻⁴ × peak`),
  excluding the IC/BC layers. A separate held-out set of 1 000 disjoint indices
  is reserved for supervised-validation MSE.

**Two-phase optimizer schedule**: 3 000 Adam steps (lr = 1e-3, gradient-clipped
at `‖∇‖ ≤ 10`) followed by 50 L-BFGS outer calls (`max_iter = 1`, strong-Wolfe
line search, `history_size = 15`). A **mandatory smoke preflight** runs first
on a mini network (3 × 64) with 1 000 collocation points and 500 Adam steps —
this catches sign errors and IC label bugs in <60 s before committing to a 17-
minute baseline run. The training profile parameters (`5 000 / 3 000 / 50`)
are a CPU-friendly compromise: a larger `10 000 / 200` schedule is
reproducible on a GPU but takes ~2 hours per case on this CPU.

The **residual implementation is validated before any training** by feeding
the analytical soliton (and separately, the analytical linear Gaussian
dispersion solution) through `PINN_NLSE.physics_residual` as a class attribute
on a fake module whose `forward` IS the analytical solution. Both cases give
max residual `≤ 1 × 10⁻⁷` — at the autograd numerical precision floor. This
test ships as `tests/test_pinn_residual.py` and is part of the `pytest tests/`
gate.

### 2.5 Pure-PINN failure and data-augmented recovery

The pure PINN (`λ_data = 0`) on the soliton case dropped the total loss four
orders of magnitude (from 0.6 to 3.4 × 10⁻⁴) over 3 000 Adam steps and reported
a textbook-looking convergence curve, yet pulse-region rel L2 vs SSFM was
**41.7 %**. Probing `|u(ξ, τ = 0)|²` along the propagation direction revealed
the failure: the network had drifted into the trivial NLSE attractor `u → 0`
(at `ξ = 5`, `|u|²` had decayed from 1.0 at `ξ = 0` to 0.53). With `λ_ic = 10`
the IC anchor is honored, but in the bulk the residual `R = i ∂_ξ u + (s/2)
∂²_τ u + N² |u|² u` is satisfied trivially when `u = 0`.

As a recovery strategy, we set `λ_data = 1.0`, added 500 SSFM supervision
points sampled from the pulse region, and reserved 1 000 disjoint held-out
indices for supervised validation MSE. All affected artifacts —
weights, history JSON, metadata JSON, loss-curve figure — are saved with the
suffix `_data_augmented_*` so the pure and data-augmented runs cannot be
silently confused. The same fallback was applied to the Gaussian case for
consistency (without it, pure-PINN rel L2 ≈ 13 %; data-augmented brings it
under the 10 % threshold).

## 3. Results
