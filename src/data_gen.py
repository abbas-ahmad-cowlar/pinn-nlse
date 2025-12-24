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

