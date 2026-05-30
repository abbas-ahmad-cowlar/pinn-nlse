# PINN-NLSE: Physics-Informed Neural Networks for Nonlinear Fiber Optics

> Solving the Nonlinear Schrödinger Equation with physics-encoded neural
> networks, benchmarked honestly against the split-step Fourier method.

**Status**: Complete. The published weights, logs, and figures are frozen for
reproducible verification.

---

## Overview

This project applies a **Physics-Informed Neural Network (PINN)** to the
**Nonlinear Schrödinger Equation (NLSE)** — the master equation governing
optical pulse propagation in single-mode fibers — and benchmarks it against a
validated **split-step Fourier method (SSFM)** ground-truth solver.

The PINN encodes the NLSE directly into its loss function via PyTorch
autograd. The complex field `u = a + ib` is split into two real outputs and
the residual `r_a, r_b` is minimized at random collocation points alongside
initial-condition, boundary-condition, and (optionally) supervised-data
losses.

The reported PINNs in this repo are **data-augmented PINNs**: pure
physics-only training on the soliton case fell into the trivial solution
`u → 0`, a documented attractor of the NLSE. We add 500 SSFM supervision
points (`λ_data = 1.0`) and label all artifacts as such, with held-out
disjoint validation MSE for honesty. The reproduction paths below describe how
to verify the published artifacts, and
[`report/technical_report.md`](report/technical_report.md) contains the full
technical discussion.

## Headline results

| Case | Pulse-region relative L2 (vs SSFM) | Notes |
|------|-----------------------------------|-------|
| N = 1 fundamental soliton | **1.29 %** | data-augmented PINN, 500 SSFM points, 1000 held-out validation labels |
| Gaussian dispersion-only | **9.29 %** | same, harder case (`s = -1`, `N² = 0`); passes the planned `< 10 %` bar with a small margin — should be read as "meets the CPU-friendly baseline target", not as a high-margin result |
| SSFM single-solve cost (CPU) | 0.10 s | reference baseline at 1024 × 1000 grid |
| PINN forward cost (CPU) | 1.20 s | on the same 1001 × 1024 evaluation grid |

For this 1D problem on CPU, the SSFM is faster than the PINN at fixed-case
inference. We report this honestly — the PINN's value lies in inverse
problems, parameter-conditioned extensions, and continuous evaluation, not
in beating SSFM at its own forward solve.

## The hero figure

The 3-panel side-by-side comparison is the project's main visual deliverable:
SSFM ground truth | PINN prediction | log₁₀ pointwise error.

![Soliton: SSFM vs PINN vs error](figures/comparison_soliton.png)

A more detailed log-scale error map and three cross-section overlays at
ξ = 0, 2.5, 5 are saved to
[`figures/error_map_soliton.png`](figures/error_map_soliton.png) and
[`figures/cross_section_soliton.png`](figures/cross_section_soliton.png).

## The physics

The normalized NLSE (Agrawal sign convention, `s = -sign(β₂)`):

$$
i\,\frac{\partial u}{\partial \xi}
\;+\; \frac{s}{2}\,\frac{\partial^2 u}{\partial \tau^2}
\;+\; N^2\,|u|^2\,u
\;=\; 0
$$

For anomalous dispersion (`s = +1`) and `N = 1`, dispersion and Kerr
nonlinearity exactly balance and the closed-form fundamental soliton is
`u(ξ, τ) = sech(τ) · exp(iξ/2)` — an exact propagation of the input pulse.
The full physics derivation — including SVEA, the co-moving frame, dispersion
length `L_D`, nonlinear length `L_NL`, soliton number `N² = L_D/L_NL`, and the
real/imaginary residual derivation — is summarized in
[`report/technical_report.md`](report/technical_report.md).

## How to reproduce

Three intended paths, with very different costs:

### Path A — Read-only verification (~30 s, or ~1 min on a fresh clone)

The fastest reproduction loads pretrained weights from `models/published/`
and the saved SSFM ground truth, then renders the comparison figures and
prints the metrics. No retraining.

```bash
git clone https://github.com/abbas-ahmad-cowlar/pinn-nlse.git
cd pinn-nlse
python -m venv venv && . venv/Scripts/activate   # Windows
# or: source venv/bin/activate                   # Unix / macOS
pip install -r requirements.txt
pip install -e .
python -m ipykernel install --user --name pinn-nlse --display-name "Python 3 (PINN-NLSE)"
jupyter notebook notebooks/03_comparison.ipynb     # Restart & Run All
```

For a pure-CLI reproduction:

```bash
python -m src.compare --case all              # regenerates comparison figures + metrics
```

> **First-clone note**: the four SSFM ground-truth `.npz` datasets
> (~46 MB total) are now whitelisted in `.gitignore` and shipped with the
> repo. If you cloned from a stripped release where they were excluded,
> `python -m src.compare --case all` and `notebooks/03_comparison.ipynb`
> will **auto-regenerate** them via `src.generate_ground_truth.generate_all()`
> (~30 s on CPU, deterministic). You can also explicitly run
> `notebooks/01_ssfm_validation.ipynb` first to regenerate them under your
> own control. Either path keeps published artifacts untouched.

For a scriptable verification run, use the read-only pytest subset in the
test-suite section below, followed by `python -m src.compare --case all` if you
want freshly rendered comparison figures.

### Path B — Smoke pipeline test (~2 min)

Sanity-check that everything wires together using a 1000-collocation × 500-
Adam-step pipeline on a mini 3 × 64 model. Final rel L2 will be ~85 % (the
model is undertrained by design); the goal is to confirm autograd, the
optimizer, and the artifact pipeline all work end-to-end.

```bash
python -m src.train --case soliton --profile smoke --run-tag smoke-test
python -m src.train --case gaussian_dispersion --profile smoke --run-tag smoke-test
```

### Path C — Full retrain on CPU (~35 min)

This reproduces the published metrics. Wall time on a 4-core CPU is **~17 min
per case**. On a GPU expect **30–60 s per case** with no code changes
(PyTorch picks up CUDA automatically).

```bash
python -m src.train --case soliton --profile baseline --data-augmented
python -m src.train --case gaussian_dispersion --profile baseline --data-augmented
python -m src.compare --case all
python -m src.benchmark            # ~6 min on CPU
```

> ⚠️ **Without `--run-tag`, Path C OVERWRITES the canonical files in
> `models/`, `logs/`, and `figures/`.** It does **NOT** touch
> `models/published/`, `logs/published/`, or `figures/published/` — those are
> frozen. The canonical files are auto-archived with a UTC timestamp before
> being overwritten (`*.archived-<UTC>.<ext>`, gitignored, recoverable on
> disk). To do a clean independent run that touches nothing else, append a
> tag:

```bash
python -m src.train --case soliton --profile baseline --data-augmented --run-tag verify-2026-05-10
```

That writes only to `models/soliton_data_augmented__verify-2026-05-10_*.pt`
and the matching log / figure paths.

## What gets written where

| Path | Written by | Published artifact behavior |
|------|------------|-------------|
| `models/published/<case>_data_augmented_final.pt` | nothing — frozen | never touched |
| `logs/published/*.json` | nothing — frozen | never touched |
| `figures/published/*.png` | nothing — frozen | never touched |
| `models/<case>_data_augmented_final.pt` | `python -m src.train --data-augmented` | overwrites with auto-archive |
| `logs/<case>_data_augmented_training_*.json` | same | overwrites with auto-archive |
| `figures/pinn_training_loss_<case>.png` | same | overwrites with auto-archive |
| `figures/comparison_*.png`, `figures/error_map_*.png`, `figures/cross_section_*.png` | `python -m src.compare` | overwrites with auto-archive |
| `figures/speed_benchmark.png`, `logs/speed_benchmark.json` | `python -m src.benchmark` | overwrites with auto-archive |
| `models/<case>_*__<tag>_*.pt` etc. | `--run-tag <tag>` runs | unique per-run |
| `*.archived-<UTC>.<ext>` | auto-saved before any overwrite | gitignored, recoverable |

## Project layout

```text
pinn-nlse/
├── README.md                          # this file
├── report/technical_report.md         # 5-page write-up
├── requirements.txt
├── pyproject.toml
├── src/
│   ├── config.py                      # physical + training-profile parameters
│   ├── utils.py                       # grid helpers, plotting, error metrics
│   ├── nlse_utils.py                  # imported from companion SSFM project
│   ├── ssfm.py                        # imported from companion SSFM project
│   ├── data_gen.py                    # collocation/IC/BC/data-point generators
│   ├── pinn_nlse.py                   # PINN model + autograd residual
│   ├── train.py                       # Adam + L-BFGS, with --run-tag and auto-archive
│   ├── generate_ground_truth.py       # regenerate the data/*.npz datasets
│   ├── compare.py                     # CLI for the comparison figures + metrics
│   └── benchmark.py                   # SSFM-vs-PINN speed benchmark
├── data/
│   ├── *.npz                          # SSFM ground-truth (regen via notebook 01)
│   └── provenance.json                # source commit + path of imported SSFM
├── models/
│   ├── published/                     # frozen canonical weights
│   └── *.pt                           # latest local training output
├── logs/
│   ├── published/                     # frozen canonical training history + metadata
│   └── *.json                         # latest local training output
├── figures/
│   ├── published/                     # frozen canonical figures
│   └── *.png                          # latest local figures and retrains
├── notebooks/
│   ├── 01_ssfm_validation.ipynb       # SSFM import + ground-truth regeneration
│   ├── 02_pinn_training.ipynb         # PINN training results + loss curves
│   └── 03_comparison.ipynb            # SSFM-vs-PINN comparison notebook
└── tests/                             # 68 pytest tests
```

## Test suite

The suite has two clearly separated subsets:

**Read-only verification subset (~15 s, no artifacts touched)**

```bash
pytest tests/test_utils.py tests/test_data_gen.py tests/test_ssfm_import.py \
       tests/test_pinn_residual.py tests/test_training_smoke.py \
       tests/test_artifact_inventory.py
```

These 63 tests load existing artifacts and check invariants (residual = 0
on analytical solutions, schema validation, presence of every required
figure/notebook/log file). Nothing on disk is rewritten.

**Full suite, including comparison-CLI side-effects (~36 s)**

```bash
pytest tests/                          # 68 tests, ~36 s on CPU
```

The 5 extra tests in `tests/test_compare_cli.py` actually invoke
`generate_soliton_comparison()` / `generate_gaussian_dispersion_comparison()`,
which **regenerates the comparison/error/cross-section figures** in
`figures/`. Thanks to the auto-archive safeguard (every overwrite renames
the prior file to `*.archived-<UTC>.png` first), no published artifact is
lost — and `figures/published/` is never touched. But the figures in
`figures/` will be rewritten with fresh timestamps. Use the read-only
subset above for strictly hands-off verification.

Test coverage:
- 8 utility tests (grid creation, sech, error metrics)
- 14 data-generator tests (collocation, IC, BC, schema, dataset/model guards)
- 7 SSFM-import tests (signature, soliton acid, energy conservation)
- 4 PINN residual tests (residual = 0 on N=1 soliton + linear Gaussian)
- 3 training smoke tests (per-term finite loss, Adam decreases loss)
- **5 comparison-CLI tests** (resolver, CLI help, end-to-end metrics) —
  these are the ones that touch `figures/`
- **27 inventory tests** (every required figure / notebook / log /
  weight file) — strictly read-only

## Key references

[1] G. P. Agrawal, *Nonlinear Fiber Optics*, 6th ed. (Academic Press, 2019).

[2] M. Raissi, P. Perdikaris, and G. E. Karniadakis, "Physics-informed neural
networks: A deep learning framework for solving forward and inverse problems
involving nonlinear partial differential equations," *Journal of
Computational Physics* 378, 686–707 (2019).

[3] X. Jiang et al., "Physics-informed Neural Network for Nonlinear Dynamics
in Fiber Optics," *Laser & Photonics Reviews* (2022); arXiv:2109.00526.

[4] J. Pu, J. Li, and Y. Chen, "Solving localized wave solutions of the
derivative nonlinear Schrödinger equation using an improved PINN method,"
arXiv:2101.08593 (2021).

## License

MIT; see [`LICENSE`](LICENSE).
