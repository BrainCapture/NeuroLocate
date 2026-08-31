# Copyright 2026 NeuroLocate contributors
# SPDX-License-Identifier: Apache-2.0
"""NeuroLocate — differentiable EEG source localization.

The outer optimization program. This package owns the JAX side: it composes the
``proposal`` and ``headfield`` Tesseracts into one differentiable function of the
source parameters, and drives it with Optax.
"""

import jax

# Float64 end to end. Both Tesseracts declare Float64 schemas, and the
# finite-difference gradient gates need the precision; JAX defaults to float32
# and would silently downcast at the boundary.
jax.config.update("jax_enable_x64", True)

__all__ = ["clients", "evaluate", "optimize", "pipeline"]
