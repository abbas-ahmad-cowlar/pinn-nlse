"""
Tests for src.utils — Phase 1 wrappers around the companion nlse_utils helpers.
"""

import numpy as np
import pytest

from src.utils import (
    compute_error_metrics,
    compute_masked_relative_l2_error,
    compute_relative_l2_error,
    create_freq_grid,
    create_time_grid,
    create_z_grid,
    sech,
)

