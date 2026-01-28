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

### 3.1 SSFM validation

The SSFM ground truth passes all three required validations (see §2.2):
soliton acid test, energy conservation, and 2nd-order Strang convergence. The
public physics-demonstration figures — soliton propagation map,
dispersion-only Gaussian broadening, SPM spectral broadening — are all produced
and saved under `figures/published/`.

| Figure | File |
|--------|------|
| Soliton propagation `|u(ξ, τ)|²` | `figures/published/gt_soliton_propagation.png` |
| Gaussian pulse broadening (`N²=0`) | `figures/published/01_dispersion_broadening.png` |
| SPM spectral broadening (`s=0`) | `figures/published/02_spm_spectral_broadening.png` |
| Strang convergence (slope ≈ 2) | `figures/published/ssfm_convergence_study.png` |

### 3.2 PINN training

Both PINNs converge cleanly under the Adam → L-BFGS schedule. The four logged
loss components (physics, IC, BC, data) all decrease monotonically except for
small spikes at the collocation-resampling steps (`resample_every = 1 000`).
Final losses:

| Case | `L_phys` (Adam end) | `L_ic` | `L_bc` | `L_data` |
|------|---------------------|--------|--------|----------|
| Soliton (data-augmented) | 4.1 × 10⁻⁵ | 0 | 1 × 10⁻⁶ | 9 × 10⁻⁶ |
| Gaussian dispersion (data-augmented) | 2.5 × 10⁻⁴ | 2 × 10⁻⁵ | 3 × 10⁻⁵ | 7 × 10⁻⁴ |

Loss-curve figures: `figures/published/pinn_training_loss_soliton.png` and
`pinn_training_loss_gaussian_dispersion.png`.

### 3.3 PINN vs SSFM comparison

The hero result is the 3-panel side-by-side comparison
(`figures/comparison_soliton.png`):

![Soliton: SSFM | PINN | log10 error](../figures/comparison_soliton.png)

Quantitative metrics are summarized in Table 1.

**Table 1**: PINN vs SSFM accuracy (data-augmented PINN, baseline profile,
seed = 42, 500 SSFM supervision points, 1 000 held-out validation points).

| Metric | Soliton (N = 1) | Gaussian dispersion |
|--------|----------------:|--------------------:|
| Pulse-region relative L2 (`|τ| ≤ 10`) | **1.29 %** | **9.29 %** |
| Full-domain relative L2 | 1.45 % | 11.29 % |
| MSE (complex field) | 1.06 × 10⁻⁵ | 5.65 × 10⁻⁴ |
| Max pointwise error | 2.69 × 10⁻² | 1.02 × 10⁻¹ |
| Mean abs error | 2.53 × 10⁻³ | 1.56 × 10⁻² |
| Held-out supervised MSE (disjoint 1 000 labels) | 2.34 × 10⁻⁵ | 8.32 × 10⁻⁴ |
| Total training time (CPU) | 1 012 s | 1 182 s |

Both pulse-region rel L2 numbers meet the plan's pass thresholds (< 5 % for
the soliton, < 10 % for the Gaussian). The held-out supervised MSE
demonstrates that the PINN generalizes to the disjoint validation labels —
not merely memorizing the 500 supervision points.

The **standalone log-scale error map** (`figures/error_map_soliton.png`) and
the **cross-section overlays at ξ = 0, 2.5, 5** (`figures/cross_section_soliton.png`)
confirm that the residual error structure is dominated by the propagation
endpoint and the wings of the pulse — the IC region and the central peak are
matched to better than `10⁻³`.

### 3.4 Speed benchmark

Figure: `figures/speed_benchmark.png`. Mode: `cpu_fair` (both SSFM and PINN
forced to CPU for an apples-to-apples comparison). Repeats per point: 3.

**Table 2**: Total wall time for `N_runs` repeated fixed-case evaluations
(median of 3 trials each). The PINN forward operates on a `1001 × 1024`
evaluation grid pre-built outside the loop; PINN end-to-end includes the
tensor-build + host-transfer overhead per call.

| `N_runs` | SSFM (s) | PINN forward (s) | PINN end-to-end (s) | fwd speedup | e2e speedup |
|---------:|---------:|-----------------:|--------------------:|------------:|------------:|
| 1 | 0.140 | 1.204 | 1.195 | 0.12× | 0.12× |
| 3 | 0.533 | 4.075 | 4.858 | 0.13× | 0.11× |
| 10 | 1.606 | 13.837 | 13.674 | 0.12× | 0.12× |
| 30 | 5.180 | 48.049 | 48.491 | 0.11× | 0.11× |

For this 1D problem on CPU, the SSFM is consistently ~8× faster than the
trained PINN at repeated fixed-case inference. This is **expected**: the SSFM
is already very efficient on small 1D grids, and the PINN forward pass over
~10⁶ collocation points is not free. We do **not** claim a parameter-sweep
speedup — that would require a parameter-conditioned PINN with `(β₂, γ, N²)`
as inputs, which is beyond this project's scope. Training time is excluded
from the timings; any PINN speed advantage would only materialize after that
one-time cost is amortized, and only in regimes where the PINN's continuous
representation, inverse-problem capability, or differentiable physics
substitutes for many SSFM solves.

## 4. Discussion

### 4.1 When PINNs help and when they don't

Honest summary of where each method dominates:

| Use case | Better tool | Why |
|----------|-------------|-----|
| Single forward simulation (1D, fixed parameters) | SSFM | Faster + machine-precision accuracy |
| Repeated fixed-case forward solves | SSFM (this 1D regime) | PINN forward over 10⁶ points still loses |
| Inverse problem (estimate `β₂, γ` from data) | PINN | Same residual loss reused; SSFM has no such tool |
| Parameter sweep over `(β₂, γ)` | Parameter-conditioned PINN (future work) | Train once, query at any parameter value |
| Continuous evaluation at arbitrary `(ξ, τ)` | PINN | Differentiable function vs discrete grid |
| Differentiable physics in a larger ML pipeline | PINN | End-to-end gradient through the PDE |

For nonlinear fiber optics research, the most directly useful extensions are
(a) parameter-conditioned PINNs for fast `(β₂, γ)` sweeps over fiber design
space and (b) inverse-problem PINNs for fitting fiber parameters to measured
pulse evolution.

### 4.2 Limitations and caveats

- **Reported PINNs are data-augmented.** The pure (no-supervision) PINN failed
  on the soliton case due to the trivial-solution attractor. Both successful
  PINNs use 500 SSFM points (`λ_data = 1.0`); all artifacts and metadata
  files state this explicitly. We deliberately preserve the pure-PINN
  artifacts in `models/`, `logs/`, and `figures/` for the honest comparison.
- **Training profile is CPU-friendly.** A larger profile with
  `N_EPOCHS_ADAM = 10 000` and `N_STEPS_LBFGS = 200` at
  `N_COLLOCATION = 5 000` is supported. We trained at `(3 000, 50, 5 000)` because the
  full schedule takes ~2 hours per case on this CPU. On a GPU, restoring the
  full schedule should improve the Gaussian case below the current 9.29 %.
- **Architecture sensitivity.** A small but real fraction of training runs
  with different seeds either took longer to converge or got stuck in
  marginally worse minima. The 5×128 architecture and Xavier init combination
  was robust; very deep (>8 layers) or very wide (>256 neurons) variants were
  less reliable on this problem.
- **The base PINN is not parameter-conditioned**. It is a single-case fixed-
  physics surrogate. Any parameter-sweep claim requires extending the input
  to include the physical parameters and training over their range.

### 4.3 Future work (6 months)

1. **Parameter-conditioned PINN** with `(β₂, γ, N²)` as additional inputs.
   This is the natural extension that would actually justify a
   parameter-sweep speedup vs SSFM.
2. **Higher-order solitons** (N = 2, 3) showing breathing dynamics. Use
   `u₀ = sech_pulse(τ)` and `N_sq = N²` per the imported solver convention
   (do **not** scale the IC by `N`).
3. **Inverse problem** demo: estimate `(β₂, γ)` from a noisy `u(z, t)`
   observation by adding the parameters as trainable variables.
4. **Method comparison**: like-for-like benchmark against DeepONet and Fourier
   Neural Operators on the same NLSE test cases.
5. **Coupled NLSEs** for birefringence and WDM channels in broader fiber-system
   modeling.
