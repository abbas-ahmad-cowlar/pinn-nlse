"""
Smoke test for the PINN training loop.

Runs `compute_total_loss` + a few Adam steps on a tiny dataset to confirm
finite-loss + decreasing-loss + gradient flow. Should complete in seconds.
"""

import numpy as np
import torch

from src.config import S_SIGN, N_SOLITON, XI_MAX, TAU_MAX, LEARNING_RATE
from src.data_gen import (