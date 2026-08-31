# Copyright 2026 NeuroLocate contributors
# SPDX-License-Identifier: Apache-2.0
"""The hybrid estimator: a learned global proposal refined through OpenMEEG.

The synthetic benchmark says plainly what limits NeuroLocate at ``K > 1``. With
distinct per-source time courses, four sources are recovered to 1.3 mm. With one
time course shared across all of them, the same physics and the same optimizer
reach 19 mm. Gradient descent through a real BEM is not the problem; the problem
is that the correlated-source objective has many deep local minima and a local
method starting from an uninformed point falls into one of them.

This package tests one hypothesis about that:

    A learned **global** proposal can put the optimizer inside the right basin,
    and continuous gradients through OpenMEEG can then refine the proposal under
    real physics. The composition should beat either half alone.

The composition is the point, and so is where each half runs:

.. code-block:: text

    EEG [C, T] + sensor geometry
        -> Tesseract `proposal`   (PyTorch, torch.autograd VJP)
        -> K continuous source positions and moments
        -> Tesseract `headfield`  (OpenMEEG C++ symmetric BEM, hand-written VJP)
        -> predicted EEG
        -> JAX / Optax loss

``jax.grad`` of that scalar runs backwards through the finite-difference position
sensitivity of a compiled C++ solver, through hand-written analytic algebra for
the moment, and into ``torch.autograd`` — with no autodiff framework in common at
any boundary. The proposal network's weights are a differentiable *input* to its
Tesseract, so the same gradient trains it.

Modules
-------
``bank``
    The training forward models. One OpenMEEG source-term assembly, re-solved
    into several plausible head models, cached as a gain bank.
``synth``
    Observations built from the bank: source sets, dynamics, noise, SNR.
``model``
    The proposal network. A coarse volumetric heatmap with continuous offsets.
``benchmark``
    The frozen hard-case matrix, in its own namespace.
"""

from __future__ import annotations

__all__: list[str] = []
