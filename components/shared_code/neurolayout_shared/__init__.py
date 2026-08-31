# Copyright 2026 NeuroLocate contributors
# SPDX-License-Identifier: Apache-2.0
"""Shared, framework-free code for the NeuroLayout Tesseracts.

Deliberately depends on NumPy only. Neither JAX nor PyTorch may be imported from
here: the whole point of the Tesseract split is that the head-physics component
and the proposal component do not share an AD stack.
"""

from .geometry import (
    fibonacci_directions,
    min_pairwise_angle,
    normalize,
    pairwise_angles,
)
from .headmodel import (
    HeadModel,
    HeadModelSpec,
    build_head_model,
    get_head_model,
    load_head_model,
    save_head_model,
)
from .sampling import (
    backward,
    forward,
    interpolation_weights,
    normalize_with_norms,
    sample_lead_field,
)
from .sphere_model import legendre_p_and_dp, sphere_lead_field

__all__ = [
    "HeadModel",
    "HeadModelSpec",
    "backward",
    "build_head_model",
    "fibonacci_directions",
    "forward",
    "get_head_model",
    "interpolation_weights",
    "legendre_p_and_dp",
    "load_head_model",
    "min_pairwise_angle",
    "normalize",
    "normalize_with_norms",
    "pairwise_angles",
    "sample_lead_field",
    "save_head_model",
    "sphere_lead_field",
]
