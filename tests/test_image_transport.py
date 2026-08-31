# Copyright 2026 NeuroLocate contributors
# SPDX-License-Identifier: Apache-2.0
"""The container-transport gate: image-backed Tesseracts must match in-process.

Everything else in the suite runs the Tesseracts via
:meth:`Tesseract.from_tesseract_api`, which imports each component into the test
process. That verifies the maths but *not* the thing Tesseract exists for — each
component running in its own image with its own dependency tree. These tests
close that gap.

They are skipped unless a Docker-compatible daemon is reachable *and* both images
have been built (``make build``), so the suite still runs on a machine without a
container runtime. The skip is deliberately loud about which precondition failed:
a silently-skipped container gate is exactly as bad as no container gate.

Run them with::

    make build && make test

Note that on a host where the invoking user was only just added to the ``docker``
group, a fresh login shell (or ``sg docker -c 'make test'``) is needed before the
daemon is reachable.
"""

from __future__ import annotations

import numpy as np
import pytest
from neurolayout.clients import IMAGE_NAMES


def _docker_unavailable() -> str | None:
    """Return a human-readable reason the image transport cannot run, else None."""
    import shutil
    import subprocess

    if shutil.which("docker") is None:
        return "docker CLI not on PATH"
    try:
        probe = subprocess.run(
            ["docker", "image", "inspect", *(f"{n}:latest" for n in IMAGE_NAMES.values())],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return f"could not invoke docker: {error}"
    if probe.returncode != 0:
        detail = (probe.stderr or probe.stdout).strip().splitlines()
        first = detail[0] if detail else "unknown error"
        if "permission denied" in first.lower():
            return f"docker daemon not reachable ({first}); try a fresh login shell"
        return f"images not built ({first}); run `make build`"
    return None


SKIP_REASON = _docker_unavailable()
pytestmark = pytest.mark.skipif(SKIP_REASON is not None, reason=str(SKIP_REASON))


#: Relative agreement demanded of the OpenMEEG path across transports.
#:
#: The OpenMEEG path contracts a [64, 4486] operator against the source term, and
#: the BLAS inside the image is not the host's, so a 4486-term dot product can
#: associate differently. That is a reduction-order difference of a few units in
#: the last place, not a code difference — but it is real, so the bar here is
#: "identical to float64 round-off" rather than "identical to the bit".
BLAS_REORDERING = 1e-13

#: The same bar for the *position* gradient, which is looser for a reason.
#:
#: That gradient is a central difference, so it divides a difference of two
#: nearly equal forward evaluations by 2h = 2e-5. A last-place discrepancy in
#: the forward is amplified by that factor before it reaches the gradient.
#: Measured across transports: ~1.5e-12 relative.
FD_AMPLIFIED_REORDERING = 1e-9


@pytest.fixture(scope="module")
def image_headfield():
    """The headfield component, served from its image."""
    from neurolayout.clients import open_component

    with open_component("headfield", "image") as tesseract:
        yield tesseract


@pytest.fixture(scope="module")
def image_pair():
    """Both components of the scientific path, each served from its own image."""
    from neurolayout.clients import HYBRID_COMPONENTS, open_components

    with open_components(HYBRID_COMPONENTS, "image") as opened:
        yield opened


def _localize_setup():
    from neurolayout.localize import Containment, LocalizeConfig, SourceParams
    from neurolayout_shared.openmeeg_model import HeadGeometry, default_artifact_path

    geometry = HeadGeometry.load(default_artifact_path())
    truth = SourceParams.from_si(
        geometry.source_space[5269], 25e-9 * geometry.source_normals[5269]
    )
    return (
        geometry,
        Containment.from_points(geometry.vertices[0]),
        LocalizeConfig(backend="openmeeg", n_times=4, steps=80),
        truth,
    )


def test_image_runs_openmeeg(image_headfield) -> None:
    """`apply` in localize mode must work inside the container."""
    from neurolayout.localize import make_waveform, predict_eeg

    _, _, config, truth = _localize_setup()
    eeg = np.asarray(predict_eeg(image_headfield, truth, make_waveform(config), config))
    assert eeg.shape == (1, 64, config.n_times)
    assert np.isfinite(eeg).all()
    # Referenced: the channel mean is zero at every time point.
    assert np.abs(eeg.sum(axis=1)).max() < 1e-12 * np.abs(eeg).max()
    # A 25 nA m dipole produces a fraction of a microvolt at the scalp.
    assert 1e-8 < np.abs(eeg).max() < 1e-5


def test_image_localize_forward_matches_local(image_headfield, tesseracts) -> None:
    """Same numbers from the container as in process, to float64 round-off."""
    from neurolayout.localize import make_waveform, predict_eeg

    _, _, config, truth = _localize_setup()
    waveform = make_waveform(config)
    from_image = np.asarray(predict_eeg(image_headfield, truth, waveform, config))
    from_local = np.asarray(predict_eeg(tesseracts.headfield, truth, waveform, config))
    scale = np.abs(from_local).max()
    assert np.abs(from_image - from_local).max() <= BLAS_REORDERING * scale


def test_image_source_position_vjp_matches_local(image_headfield, tesseracts) -> None:
    """The finite differences really are being taken inside the container."""
    rng = np.random.default_rng(31)
    inputs = {
        "mode": "localize",
        "backend": "openmeeg",
        "source_positions": np.array([[-0.037, 0.023, 0.036]]),
        "source_timecourses": 2e-8 * rng.standard_normal((1, 1, 3, 4)),
    }
    cotangent = {"eeg": rng.standard_normal((1, 64, 4))}
    grads = [
        tesseract.vector_jacobian_product(
            inputs=inputs,
            vjp_inputs=["source_positions", "source_timecourses"],
            vjp_outputs=["eeg"],
            cotangent_vector=cotangent,
        )
        for tesseract in (image_headfield, tesseracts.headfield)
    ]
    for name in ("source_positions", "source_timecourses"):
        local = np.asarray(grads[1][name])
        difference = np.abs(np.asarray(grads[0][name]) - local).max()
        tolerance = (
            FD_AMPLIFIED_REORDERING if name == "source_positions" else BLAS_REORDERING
        )
        assert difference <= tolerance * np.abs(local).max(), (
            f"{name} differs by {difference:.3e}"
        )
    assert np.abs(np.asarray(grads[0]["source_positions"])).max() > 0


def test_image_backed_localization_recovers_the_source(image_headfield) -> None:
    """End to end: JAX drives the containerized BEM to the right answer."""
    from neurolayout.localize import (
        METRES_PER_CM,
        least_squares_moment,
        run_localization,
        simulate,
    )

    geometry, containment, config, truth = _localize_setup()
    observation = simulate(image_headfield, truth, config)
    start = least_squares_moment(
        image_headfield,
        np.asarray(truth.position_cm)
        + (geometry.vertices[0].mean(axis=0) - geometry.source_space[5269]) / METRES_PER_CM * 0.3,
        observation,
        config,
    )
    result = run_localization(
        image_headfield, observation, start, config, containment, record_every=40
    )
    assert result["initial_error_mm"] > 10.0
    assert result["final_error_mm"] < 1.0


#
# The two-component path through images
#
# This is what the repository is a demonstration of: one `jax.grad` whose
# evaluation enters a container running PyTorch, then a container running
# OpenMEEG's C++ BEM, and comes back with a gradient with respect to the
# network's weights. Nothing in the calling process imports either framework.
#


def _composed_problem(headfield, proposal):
    """A K=2 problem on the real 64-channel head model, and the composed loss."""
    import jax.numpy as jnp
    from neurolayout.clients import COMPONENT_ROOT
    from neurolayout.hybrid.model import flat_parameters, load_checkpoint
    from neurolayout.hybrid.physics import make_physics_loss
    from neurolayout.localize import Containment, LocalizeConfig
    from neurolayout_shared.openmeeg_model import HeadGeometry, default_artifact_path

    geometry = HeadGeometry.load(default_artifact_path())
    containment = Containment.from_points(geometry.vertices[0])
    config = LocalizeConfig(backend="openmeeg", n_times=8, sfreq=160.0)

    rng = np.random.default_rng(4)
    eeg = rng.normal(0.0, 1e-6, (1, 64, 8))
    eeg -= eeg.mean(axis=1, keepdims=True)

    # The same checkpoint the image packages, read here only to recover the
    # parameter vector the component expects as its differentiable input.
    model, _ = load_checkpoint(COMPONENT_ROOT / "proposal" / "proposal.pt")
    weights = jnp.asarray(flat_parameters(model).numpy())

    loss = make_physics_loss(
        headfield, proposal, config, containment, n_sources=2
    )
    return loss, weights, jnp.asarray(eeg), jnp.ones((1, 64))


def test_both_images_serve_the_composed_forward(image_pair) -> None:
    """The composed loss must evaluate with both components in containers."""
    loss, weights, eeg, mask = _composed_problem(
        image_pair["headfield"], image_pair["proposal"]
    )
    assert np.isfinite(float(loss(weights, eeg, mask)))


def test_composed_image_gradient_is_finite_and_non_zero(image_pair) -> None:
    """One `jax.grad`, two containers, two derivative mechanisms, one cotangent."""
    import jax

    loss, weights, eeg, mask = _composed_problem(
        image_pair["headfield"], image_pair["proposal"]
    )
    value, gradient = jax.value_and_grad(loss)(weights, eeg, mask)
    assert np.isfinite(float(value))
    gradient = np.asarray(gradient)
    assert gradient.shape == weights.shape
    assert np.isfinite(gradient).all()
    assert np.abs(gradient).max() > 0.0
