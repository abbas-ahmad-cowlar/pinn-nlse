# Published model weights

This directory holds the **frozen, canonical PINN weights** used in the
technical report and notebook. **Do not run training into this directory.**

The training code in `src/train.py` only writes to `models/` (and the optional
tagged paths under `models/<case>__<run_tag>_*`). It never writes to
`models/published/`.

**Contents**:

- `soliton_data_augmented_final.pt` - PINN for the N=1 soliton, trained with
  500 SSFM supervision points on top of the physics-informed loss
  (`lambda_data = 1.0`). Pulse-region relative L2 error vs SSFM: **1.29 %**.
- `gaussian_dispersion_data_augmented_final.pt` - PINN for the linear
  Gaussian-dispersion case (`s = -1`, `N_sq = 0`), data-augmented identically.
  Pulse-region relative L2 error vs SSFM: **9.29 %**.

The corresponding training history, metadata, and loss-curve figures live in
`logs/published/` and `figures/published/`.

If you want to do an independent training run, use `--run-tag <YOUR_TAG>` so
your run lands at `models/<case>__<YOUR_TAG>_final.pt` and never touches the
files here.