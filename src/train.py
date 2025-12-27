"""
PINN-NLSE Training Script
==========================

Trains the PINN model on the NLSE using a two-stage optimizer strategy:
  Stage 1: Adam (coarse convergence)
  Stage 2: L-BFGS (fine convergence; bounded full-batch outer calls)

Loss = lambda_phys * L_physics + lambda_ic * L_IC + lambda_bc * L_BC + lambda_data * L_data

"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Callable, Optional

import numpy as np
import torch

# Imports moved into wrappers / main to keep module-level cost light:
#   from src.pinn_nlse import PINN_NLSE
#   from src.config import ...
#   from src.data_gen import ...
#   from src.utils import ...


# ---------------------------------------------------------------------------
# Loss + helpers