"""
PINN Training Data Generators
==============================

Generates the four types of training points for the PINN-NLSE model:
1. Collocation points: random (xi, tau) for physics loss (no labels)
2. Initial condition points: (xi=0, tau) with labels from u_0
3. Boundary condition points: (xi, tau=+/-tau_max) with labels = 0
4. Data points (optional): (xi, tau) with labels from SSFM solution

All outputs are PyTorch tensors with requires_grad as needed.

"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Generator 1: Collocation points (physics loss, no labels)
# ---------------------------------------------------------------------------

def generate_collocation_points(N_coll: int, xi_max: float, tau_max: float,
                                device: str = "cpu",
                                seed: Optional[int] = None,
                                method: str = "random"):
    """
    Generate random collocation points for physics loss.

    These points have NO labels — the loss is the PDE residual computed from
    the PINN's own output via autograd. requires_grad=True so PyTorch can
    differentiate through xi and tau when computing the residual.

    Args:
        N_coll: Number of collocation points
        xi_max: Max propagation distance
        tau_max: Max time (half-width)
        device: PyTorch device ('cpu' or 'cuda')
        seed: Optional seed for reproducible sampling
        method: 'random' or 'sobol' low-discrepancy sampling

    Returns:
        xi_coll: Tensor (N_coll, 1), requires_grad=True
        tau_coll: Tensor (N_coll, 1), requires_grad=True
    """
    if method == "random":
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed))
        xi_unit = torch.rand(N_coll, 1, device=device, generator=generator)
        tau_unit = torch.rand(N_coll, 1, device=device, generator=generator)
    elif method == "sobol":
        # SobolEngine seeds via the seed argument; if None it auto-randomizes.
        engine = torch.quasirandom.SobolEngine(
            dimension=2, scramble=True,
            seed=int(seed) if seed is not None else None,
        )
        pts = engine.draw(N_coll).to(device)
        xi_unit = pts[:, 0:1]
        tau_unit = pts[:, 1:2]
    else:
        raise ValueError(f"Unknown collocation sampling method: {method}")

    xi = xi_unit * xi_max
    tau = (tau_unit * 2 - 1) * tau_max
    xi.requires_grad_(True)
    tau.requires_grad_(True)
    return xi, tau


# ---------------------------------------------------------------------------
# Generator 2: Initial condition points (xi = 0)
# ---------------------------------------------------------------------------

def generate_ic_points(N_ic: int, tau_max: float, ic_func: str = "sech",
                       device: str = "cpu"):
    """
    Generate initial condition points at xi = 0.

    Uses the SSFM/FFT-grid convention: includes -tau_max, excludes +tau_max.

    Args:
        N_ic: Number of IC points
        tau_max: Max time (half-width)
        ic_func: Initial condition type ('sech' or 'gaussian')
        device: PyTorch device

    Returns:
        xi_ic: Tensor (N_ic, 1), all zeros (xi = 0)
        tau_ic: Tensor (N_ic, 1), uniformly spaced on [-tau_max, +tau_max)
        a_ic: Tensor (N_ic, 1), real part of u_0
        b_ic: Tensor (N_ic, 1), imaginary part of u_0
    """
    xi = torch.zeros(N_ic, 1, device=device)
    # Match SSFM FFT grid convention: include -tau_max, exclude +tau_max.
    tau = torch.linspace(-tau_max, tau_max, N_ic + 1, device=device)[:-1].unsqueeze(1)

    if ic_func == "sech":
        a = 1.0 / torch.cosh(tau)
        b = torch.zeros_like(tau)
    elif ic_func == "gaussian":
        a = torch.exp(-tau ** 2 / 2)
        b = torch.zeros_like(tau)
    else:
        raise ValueError(f"Unknown IC function: {ic_func}")

    return xi, tau, a, b


# ---------------------------------------------------------------------------
# Generator 3: Boundary condition points (tau = +/- tau_max)
# ---------------------------------------------------------------------------

def generate_bc_points(N_bc: int, xi_max: float, tau_max: float,
                       device: str = "cpu"):
    """
    Generate boundary condition points at tau = +/- tau_max.

    Coordinate note: +tau_max is an analytical far-field point, not a stored
    SSFM FFT-grid point. The grid itself excludes +tau_max; this is checked
    separately in assert_boundary_decay() for any data we want to use as
    zero-BC training labels.

    The BC enforces u -> 0 at the temporal boundaries. Because sech(20) ~ 4e-9
    and Gaussian(20) ~ 1e-87, this is effectively a Dirichlet zero condition.

    Args:
        N_bc: Number of BC points (split equally between -tau_max and +tau_max)
        xi_max: Max propagation distance
        tau_max: Max time (half-width)
        device: PyTorch device

    Returns:
        xi_bc: Tensor (N_bc, 1), random xi values in [0, xi_max]
        tau_bc: Tensor (N_bc, 1), values at -tau_max or +tau_max
        a_bc: Tensor (N_bc, 1), all zeros (target)
        b_bc: Tensor (N_bc, 1), all zeros (target)
    """
    N_half = N_bc // 2
    xi = torch.rand(N_bc, 1, device=device) * xi_max

    tau_left = -tau_max * torch.ones(N_half, 1, device=device)
    tau_right = tau_max * torch.ones(N_bc - N_half, 1, device=device)
    tau = torch.cat([tau_left, tau_right], dim=0)

    a = torch.zeros(N_bc, 1, device=device)
    b = torch.zeros(N_bc, 1, device=device)

    return xi, tau, a, b


# ---------------------------------------------------------------------------
# Generator 4: Data points from SSFM ground truth
# ---------------------------------------------------------------------------

def generate_data_points(u_hist: np.ndarray, xi_arr: np.ndarray, tau: np.ndarray,
                         N_data: int,
                         device: str = "cpu",
                         sampling: str = "pulse_region",
                         tau_abs_max: float = 10.0,
                         intensity_floor: float = 1e-4,
                         exclude_initial_boundary: bool = True,
                         rng: Optional[np.random.Generator] = None,
                         seed: Optional[int] = None,
                         exclude_flat_indices: Optional[Iterable[int]] = None,
                         return_indices: bool = False):
    """
    Sample labeled data points from SSFM ground truth.

    Samples N_data points from the SSFM solution grid and returns their
    (xi, tau, Re(u), Im(u)) as PyTorch tensors. The default focuses
    supervision on the physical pulse instead of spending most labels in
    low-amplitude window tails.

    Args:
        u_hist: Complex array (N_z+1, N_t) from SSFM
        xi_arr: Distance array (N_z+1,) from SSFM
        tau: Time array (N_t,) from SSFM
        N_data: Number of data points to sample
        device: PyTorch device
        sampling: 'uniform' or 'pulse_region'
        tau_abs_max: Keep |tau| <= tau_abs_max when sampling='pulse_region'
        intensity_floor: Keep points with |u|^2 >= intensity_floor * max(|u|^2)
        exclude_initial_boundary: Avoid duplicating IC/BC losses in data points
        rng: Optional np.random.Generator for reproducible sampling
        seed: Optional seed used only when rng is not supplied
        exclude_flat_indices: Optional iterable of flat SSFM-grid indices to remove
            from the candidate pool. Use this to create held-out supervised
            validation points disjoint from data-augmented training labels.
        return_indices: If True, also return the sampled flat SSFM-grid indices.

    Returns:
        xi_data: Tensor (N_data, 1)
        tau_data: Tensor (N_data, 1)
        a_data: Tensor (N_data, 1), Re(u_ssfm)
        b_data: Tensor (N_data, 1), Im(u_ssfm)
        flat_idx: Optional NumPy array (N_data,) when return_indices=True
    """
    N_z_plus_1, N_t = u_hist.shape
    candidate_mask = np.ones((N_z_plus_1, N_t), dtype=bool)

    if sampling == "pulse_region":
        candidate_mask &= np.abs(tau)[None, :] <= tau_abs_max
        if intensity_floor is not None:
            intensity = np.abs(u_hist) ** 2
            candidate_mask &= intensity >= intensity_floor * np.max(intensity)
    elif sampling != "uniform":
        raise ValueError(f"Unknown sampling strategy: {sampling}")

    if exclude_initial_boundary:
        candidate_mask[0, :] = False     # exclude xi=0 (IC layer)
        candidate_mask[:, 0] = False     # exclude tau = -tau_max (BC layer)
        candidate_mask[:, -1] = False    # exclude tau = last grid point (BC-adjacent)

    candidates = np.flatnonzero(candidate_mask.ravel())
    if exclude_flat_indices is not None:
        blocked = np.asarray(list(exclude_flat_indices), dtype=np.int64)
        candidates = np.setdiff1d(candidates, blocked, assume_unique=False)

    if len(candidates) == 0:
        raise ValueError("No candidate SSFM data points available; relax sampling filters")

    if rng is None:
        rng = np.random.default_rng(seed)

    replace = N_data > len(candidates)
    flat_idx = rng.choice(candidates, size=N_data, replace=replace)
    idx_z, idx_t = np.unravel_index(flat_idx, u_hist.shape)

    xi_vals = xi_arr[idx_z]
    tau_vals = tau[idx_t]
    u_vals = u_hist[idx_z, idx_t]

    xi_data = torch.tensor(xi_vals, dtype=torch.float32, device=device).unsqueeze(1)
    tau_data = torch.tensor(tau_vals, dtype=torch.float32, device=device).unsqueeze(1)
    a_data = torch.tensor(np.real(u_vals), dtype=torch.float32, device=device).unsqueeze(1)
    b_data = torch.tensor(np.imag(u_vals), dtype=torch.float32, device=device).unsqueeze(1)

    if return_indices:
        return xi_data, tau_data, a_data, b_data, flat_idx
    return xi_data, tau_data, a_data, b_data


# ---------------------------------------------------------------------------
# Helpers: ground-truth schema validation and dataset/model guards
# ---------------------------------------------------------------------------

REQUIRED_NPZ_KEYS = (
    "schema_version", "case_name", "tau", "xi", "omega", "u_hist",
    "dtau", "dxi", "N_T", "N_Z", "tau_max", "xi_max",
    "s", "N_sq", "ic_type", "case_type", "dispersion_disabled", "equation",
    "u_dtype", "tau_dtype", "max_boundary_intensity", "generated_at_utc",
    "pinn_repo_commit", "ssfm_source_commit", "ssfm_source_path",
)


def load_ground_truth_npz(path: str):
    """
    Load and validate a ground-truth .npz file created by the SSFM generator.

    Returns the loaded NumPy archive after checking the schema, coordinate
    lengths, and u_hist shape. Tensor conversion happens in the data
    generator functions, not here.
    """
    data = np.load(path, allow_pickle=False)

    for key in REQUIRED_NPZ_KEYS:
        if key not in data:
            raise KeyError(f"{path} missing required key: {key}")

    if data["u_hist"].shape != (len(data["xi"]), len(data["tau"])):
        raise ValueError(
            f"{path} has u_hist shape {data['u_hist'].shape}, "
            f"expected ({len(data['xi'])}, {len(data['tau'])})"
        )

    if data["omega"].shape != data["tau"].shape:
        raise ValueError(f"{path} omega shape must match tau shape")

    if int(data["N_T"]) != len(data["tau"]) or int(data["N_Z"]) != len(data["xi"]) - 1:
        raise ValueError(f"{path} grid metadata does not match stored arrays")

    if not np.iscomplexobj(data["u_hist"]):
        raise TypeError(f"{path} u_hist must be complex-valued")

    if float(np.asarray(data["max_boundary_intensity"])) >= 1e-6:
        raise ValueError(
            f"{path} boundary leakage is too large for zero temporal BCs: "
            f"{float(np.asarray(data['max_boundary_intensity'])):.2e}; "
            "increase TAU_MAX and regenerate the dataset before using zero temporal BCs"
        )

    return data


def assert_boundary_decay(data, tolerance: float = 1e-6) -> float:
    """
    Confirm that a dataset is compatible with zero temporal BCs.

    SSFM is FFT-periodic, while the PINN uses u -> 0 at tau boundaries; this
    check makes that approximation explicit. Zero Dirichlet temporal BCs are
    accepted only when both the stored edge intensity and the analytical
    far-field tail intensity are below `tolerance`. Failures are blocking —
    increase TAU_MAX and regenerate before training.
    """
    edge = np.abs(np.concatenate(
        [data["u_hist"][:, :2], data["u_hist"][:, -2:]], axis=1
    )) ** 2
    max_edge = float(np.max(edge))

    tau_max = float(np.asarray(data["tau_max"]))
    ic_type = str(np.asarray(data["ic_type"]))
    if ic_type == "sech":
        tail_amp = 2.0 * np.exp(-tau_max) / (1.0 + np.exp(-2.0 * tau_max))
    elif ic_type == "gaussian":
        tail_amp = np.exp(-0.5 * tau_max ** 2)
    else:
        tail_amp = max(abs(data["u_hist"][0, 0]), abs(data["u_hist"][0, -1]))
    max_analytic_tail_intensity = float(abs(tail_amp) ** 2)

    case_name = str(np.asarray(data["case_name"]))
    if max_edge >= tolerance:
        raise ValueError(
            f"{case_name}: max boundary intensity {max_edge:.2e} >= "
            f"{tolerance:.1e}; increase TAU_MAX and regenerate before "
            "using zero temporal BCs"
        )
    if max_analytic_tail_intensity >= tolerance:
        raise ValueError(
            f"{case_name}: analytical +TAU_MAX tail intensity "
            f"{max_analytic_tail_intensity:.2e} >= {tolerance:.1e}; increase TAU_MAX"
        )
    return max(max_edge, max_analytic_tail_intensity)


def assert_case_matches_model(data, model, expected_ic_type: Optional[str] = None) -> bool:
    """
    Fail fast if a ground-truth dataset does not match the PINN physics.

    Prevents silently comparing or supervising a model trained with one
    sign/nonlinearity against SSFM data generated with another.
    """
    s_data = float(np.asarray(data["s"]))
    N_sq_data = float(np.asarray(data["N_sq"]))
    s_model = float(model.s)
    N_sq_model = float(model.N_sq)

    if not np.isclose(s_data, s_model):
        raise ValueError(f"Dataset/model sign mismatch: data s={s_data}, model s={s_model}")

    if not np.isclose(N_sq_data, N_sq_model):
        raise ValueError(
            f"Dataset/model nonlinearity mismatch: "
            f"data N_sq={N_sq_data}, model N_sq={N_sq_model}"
        )

    if expected_ic_type is not None:
        ic_type = str(np.asarray(data["ic_type"]))
        if ic_type != expected_ic_type:
            raise ValueError(
                f"Dataset IC mismatch: data ic_type={ic_type}, expected {expected_ic_type}"
            )

    return True
