"""
Configuration Module - Physical & PINN Hyperparameters
=======================================================

Centralizes ALL parameters for the PINN-NLSE project.
Imported by every other module and notebook.

Sign Convention: Agrawal (Nonlinear Fiber Optics, 6th ed.)
    s = -sign(beta_2) = +1 for anomalous dispersion (soliton regime)
    Normalized NLSE: i d_xi u + (s/2) d_tau^2 u + N^2 |u|^2 u = 0

"""

import numpy as np

# ==============================================================
# GROUP 1: Physical Parameters (Normalized Units)
# ==============================================================

N_SOLITON = 1        # Soliton order N. N=1: fundamental soliton (sech propagates unchanged).
                     # N=2: second-order soliton (periodic breathing). N^2 = L_D / L_NL.

S_SIGN = 1           # Sign parameter: s = -sign(beta_2).
                     # s = +1 -> anomalous dispersion (beta_2 < 0) -> solitons possible
                     # s = -1 -> normal dispersion (beta_2 > 0) -> pulse broadening only

# ==============================================================
# GROUP 2: Numerical Grid Parameters
# ==============================================================

N_T = 1024           # Time grid points. Power of 2 for fast, predictable FFT performance.
                     # Modern FFTs handle other sizes, but 1024 = 2^10 is efficient.

N_Z = 1000           # Number of propagation steps (SSFM). More = more accurate.
                     # d_xi = XI_MAX / N_Z. For N_Z=1000, XI_MAX=5: d_xi = 0.005.

TAU_MAX = 20.0       # Half-width of time window: tau in [-TAU_MAX, +TAU_MAX].
                     # Must be wide enough that the pulse decays to ~0 at boundaries.
                     # sech(20) ~ 4.1e-9, and the Gaussian dispersion case at xi=5
                     # remains safely inside the window.

XI_MAX = 5.0         # Total propagation distance in units of L_D.
                     # For N=1, the intensity profile is invariant at every xi.
                     # pi/2 is a higher-order soliton breathing scale, not an N=1 recurrence.

# ==============================================================
# GROUP 3: PINN Hyperparameters
# ==============================================================

# --- Network Architecture ---
N_HIDDEN_LAYERS = 5  # Number of hidden layers. 4-6 is typical for PINNs.
N_NEURONS = 128      # Neurons per hidden layer. 64-256 is typical.
ACTIVATION = "tanh"  # MUST be C-infinity (smooth). tanh recommended.
                     # DO NOT use ReLU - discontinuous 2nd derivative breaks autograd.

# --- Training Schedule ---
LEARNING_RATE = 1e-3 # Initial learning rate for Adam optimizer.
N_EPOCHS_ADAM = 20000   # Adam optimizer steps (coarse convergence).
N_STEPS_LBFGS = 500     # Bounded PyTorch L-BFGS outer calls; max_iter=1 per call by default.
N_EPOCHS_TOTAL = N_EPOCHS_ADAM + N_STEPS_LBFGS  # Total outer optimization steps.

# --- Collocation & Training Points ---
N_COLLOCATION = 20000   # Random (xi, tau) points for physics loss. 10K-50K typical.
N_IC_POINTS = 500       # Points along tau at xi=0 for initial condition loss.
N_BC_POINTS = 200       # Points along xi at tau = +/- TAU_MAX for boundary condition loss.

# --- Loss Weights ---
LAMBDA_PHYSICS = 1.0    # Weight for PDE residual loss.
LAMBDA_IC = 10.0        # Weight for initial condition loss (high = stronger enforcement).
LAMBDA_BC = 1.0         # Weight for boundary condition loss.
LAMBDA_DATA = 0.0       # Weight for SSFM data supervision. 0.0 = pure PINN (no data).
                        # Set to 1.0 and provide SSFM data points if accuracy is poor.

# --- Logging ---
LOG_EVERY = 100         # Print loss every N optimizer steps/outer calls.
CHECKPOINT_EVERY = 5000 # Save model checkpoint every N Adam steps.

FIGURE_PATHS = {
    "gt_soliton_propagation": "figures/gt_soliton_propagation.png",
    "ssfm_verification_soliton": "figures/ssfm_verification_soliton.png",
    "dispersion_broadening": "figures/01_dispersion_broadening.png",
    "dispersion_width": "figures/01_dispersion_width_vs_xi.png",
    "spm_spectral_broadening": "figures/02_spm_spectral_broadening.png",
    "gt_gaussian_nonlinear": "figures/gt_gaussian_nonlinear_propagation.png",
    "comparison_soliton": "figures/comparison_soliton.png",
    "error_map_soliton": "figures/error_map_soliton.png",
    "cross_section_soliton": "figures/cross_section_soliton.png",
    "comparison_gaussian_dispersion": "figures/comparison_gaussian_dispersion.png",
    "error_map_gaussian_dispersion": "figures/error_map_gaussian_dispersion.png",
    "cross_section_gaussian_dispersion": "figures/cross_section_gaussian_dispersion.png",
    "training_loss_soliton": "figures/pinn_training_loss_soliton.png",
    "training_loss_gaussian_dispersion": "figures/pinn_training_loss_gaussian_dispersion.png",
    "speed_benchmark": "figures/speed_benchmark.png",
}

TRAINING_PROFILES = {
    "smoke": {
        "N_COLLOCATION": 1000,
        "N_IC_POINTS": 100,
        "N_BC_POINTS": 50,
        "N_EPOCHS_ADAM": 500,
        "N_STEPS_LBFGS": 0,
        "LOG_EVERY": 50,
        "LBFGS_HISTORY_SIZE": 10,
        "LBFGS_MAX_COLLOCATION": 1000,
        "RESAMPLE_EVERY": None,
    },
    "baseline": {