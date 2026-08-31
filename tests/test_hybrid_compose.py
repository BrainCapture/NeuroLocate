# Copyright 2026 NeuroLocate contributors
# SPDX-License-Identifier: Apache-2.0
r"""Gate 2 for the hybrid estimator: gradients across *both* new boundaries.

The claim under test is the whole point of the package, so it is tested the only
way a claim like this can be — against central differences of the same composed
forward, with no shared code between the two sides:

.. code-block:: text

    loss
      -> headfield  VJP : central differences through the volume-conduction solver
      -> source positions and moments
      -> proposal   VJP : torch.autograd through the network
      -> network parameters

If ``test_composed_weight_gradient_matches_finite_differences`` passes, then
``dL/dweights`` really does travel back through the physics component, and a
training step on it is a training step through the solver. If it fails, every
result downstream is a result about something else.

The sphere backend is used for the fast tests: which volume-conduction solver
sits inside ``headfield`` is not what these tests are about, and the sphere
backend needs no cached artifact and runs in milliseconds. The OpenMEEG backend
gets its own test, skipped when the artifact is absent.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("torch")

import torch  # noqa: E402
from neurolayout.clients import open_component  # noqa: E402
from neurolayout.hybrid.model import (  # noqa: E402
    ProposalConfig,
    ProposalNet,
    save_checkpoint,
)
from neurolayout.hybrid.physics import (  # noqa: E402
    columns,
    containment_penalty,
    make_physics_loss,
    projector_residual,
    proposal_outputs,
)
from neurolayout.localize import Containment, LocalizeConfig  # noqa: E402
from neurolayout_shared.openmeeg_model import (  # noqa: E402
    HeadGeometry,
    default_artifact_path,
)

N_CHANNELS = 24
N_TIMES = 8
N_SOURCES = 2
BATCH = 2

#: A sphere-backend problem that needs no cached BEM artifact.
SPHERE_CONFIG = LocalizeConfig(
    backend="sphere", n_times=N_TIMES, n_channels=N_CHANNELS
)
SPHERE_CONTAINMENT = Containment(centre_cm=np.zeros(3), semi_axes_cm=np.full(3, 7.0))


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory, monkeypatch_session):
    """A deterministic miniature proposal network, staged where the component looks.

    Deliberately tiny — eight voxels, width 16 — because these tests are about the
    derivative bookkeeping, and a full-sized network would make the
    finite-difference sweep slow without making it more convincing.
    """
    directory = tmp_path_factory.mktemp("proposal")
    torch.manual_seed(0)
    centres = np.array(
        [[x, y, z] for x in (-0.03, 0.03) for y in (-0.03, 0.03) for z in (-0.02, 0.02)]
    )
    sensors = _sphere_sensors(N_CHANNELS)
    model = ProposalNet(
        ProposalConfig(
            n_channels=N_CHANNELS,
            width=16,
            depth=1,
            heads=2,
            voxel_dim=8,
            max_sources=N_SOURCES,
            grid_pitch_m=0.02,
        ),
        centres,
        sensors,
    )
    # A zero-initialized offset head has zero gradient through the position at
    # step 0 only because tanh'(0) = 1 times a zero weight; perturb it so the
    # finite-difference check exercises a generic point rather than a special one.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.05 * torch.randn_like(parameter))
    save_checkpoint(directory / "tiny.pt", model, {"purpose": "tests"})
    monkeypatch_session.setenv("NEUROLOCATE_PROPOSAL_DIR", str(directory))
    return "tiny"


@pytest.fixture(scope="session")
def monkeypatch_session():
    """A module-scoped monkeypatch, which pytest does not provide by default."""
    patch = pytest.MonkeyPatch()
    yield patch
    patch.undo()


def _sphere_sensors(n_channels: int) -> np.ndarray:
    """The sphere backend's own electrode array, so the two agree on geometry."""
    from neurolayout_shared.sphere_model import SphereHead

    return SphereHead(radius=0.09, sigma=0.33, n_channels=n_channels).sensor_xyz()


@pytest.fixture(scope="module")
def proposal(tiny_checkpoint):
    with open_component("proposal", "local") as tesseract:
        yield tesseract


@pytest.fixture(scope="module")
def headfield():
    with open_component("headfield", "local") as tesseract:
        yield tesseract


@pytest.fixture(scope="module")
def observation():
    """A deterministic epoch and an all-ones channel mask."""
    rng = np.random.default_rng(3)
    eeg = rng.normal(0.0, 1e-6, (BATCH, N_CHANNELS, N_TIMES))
    eeg -= eeg.mean(axis=1, keepdims=True)
    return jnp.asarray(eeg), jnp.ones((BATCH, N_CHANNELS))


@pytest.fixture(scope="module")
def weights(tiny_checkpoint):
    """The checkpoint's own parameters, flattened, as the differentiable input."""
    import os

    from neurolayout.hybrid.model import flat_parameters, load_checkpoint

    model, _ = load_checkpoint(
        f"{os.environ['NEUROLOCATE_PROPOSAL_DIR']}/tiny.pt"
    )
    return jnp.asarray(flat_parameters(model).numpy())


#
# The pieces, before the composition
#


def test_the_proposal_component_returns_a_source_set(
    proposal, observation, weights, tiny_checkpoint
):
    """Gate A for the new component: schema-valid output at the right shape."""
    eeg, mask = observation
    outputs = proposal.apply(
        {
            "eeg": np.asarray(eeg),
            "channel_mask": np.asarray(mask),
            "weights": np.asarray(weights),
            "checkpoint": tiny_checkpoint,
            "n_sources": N_SOURCES,
        }
    )
    assert np.asarray(outputs["positions_m"]).shape == (BATCH, N_SOURCES, 3)
    assert np.asarray(outputs["moments"]).shape == (BATCH, N_SOURCES, 3)
    assert np.isfinite(np.asarray(outputs["positions_m"])).all()
    directions = np.asarray(outputs["moments"])
    np.testing.assert_allclose(np.linalg.norm(directions, axis=2), 1.0, atol=1e-9)


def test_the_proposal_does_not_return_two_of_the_same_source(
    proposal, observation, weights, tiny_checkpoint
):
    """Non-maximum suppression is what stops one blob being read as K sources."""
    eeg, mask = observation
    outputs = proposal.apply(
        {
            "eeg": np.asarray(eeg),
            "channel_mask": np.asarray(mask),
            "weights": np.asarray(weights),
            "checkpoint": tiny_checkpoint,
            "n_sources": N_SOURCES,
            "nms_radius_m": 0.010,
        }
    )
    positions = np.asarray(outputs["positions_m"])
    for entry in positions:
        separation = np.linalg.norm(entry[0] - entry[1])
        assert separation > 0.0


def test_the_columns_are_the_per_source_topographies(headfield, observation):
    """``eeg[:, :, k]`` under the delta construction is exactly ``G(p_k) m_k``.

    Checked against the component's ordinary ``localize`` mode, which knows
    nothing about the construction: one source at a time, one sample, unit time
    course.
    """
    rng = np.random.default_rng(5)
    positions = jnp.asarray(rng.uniform(-0.04, 0.04, (BATCH, N_SOURCES, 3)))
    moments = jnp.asarray(rng.normal(0.0, 1e-8, (BATCH, N_SOURCES, 3)))
    built = np.asarray(columns(headfield, positions, moments, SPHERE_CONFIG))
    assert built.shape == (BATCH, N_CHANNELS, N_SOURCES)

    for b in range(BATCH):
        for k in range(N_SOURCES):
            single = headfield.apply(
                {
                    "source_positions": np.asarray(positions[b, k])[None],
                    "source_timecourses": np.asarray(moments[b, k])[
                        None, None, :, None
                    ],
                    **SPHERE_CONFIG.static_inputs(),
                }
            )
            np.testing.assert_allclose(
                built[b, :, k], np.asarray(single["eeg"])[0, :, 0], rtol=1e-12
            )


def test_the_projector_residual_is_zero_on_data_it_can_explain():
    """A residual that is not zero when the columns span the data is not a residual."""
    rng = np.random.default_rng(7)
    column_matrix = jnp.asarray(rng.normal(size=(1, N_CHANNELS, N_SOURCES)))
    amplitudes = rng.normal(size=(1, N_SOURCES, N_TIMES))
    observed = jnp.asarray(np.einsum("bck,bkt->bct", np.asarray(column_matrix), amplitudes))
    assert float(projector_residual(column_matrix, observed)[0]) < 1e-12


def test_the_projector_residual_is_one_on_data_orthogonal_to_the_columns():
    """And the other end of the scale, so the number has a readable meaning."""
    column_matrix = jnp.zeros((1, N_CHANNELS, N_SOURCES)).at[0, 0, 0].set(1.0)
    observed = jnp.zeros((1, N_CHANNELS, N_TIMES)).at[0, 1, :].set(1.0)
    np.testing.assert_allclose(
        float(projector_residual(column_matrix, observed)[0]), 1.0, atol=1e-9
    )


def test_containment_is_exactly_zero_inside_the_ellipsoid():
    """A penalty that is active at the answer would be shaping the answer."""
    inside = jnp.zeros((1, N_SOURCES, 3))
    assert float(containment_penalty(inside, SPHERE_CONTAINMENT)[0]) == 0.0
    outside = jnp.asarray(np.full((1, N_SOURCES, 3), 0.2))
    assert float(containment_penalty(outside, SPHERE_CONTAINMENT)[0]) > 0.0


#
# Gate 2: the composed gradient, across both boundaries
#


def _loss_fn(headfield, proposal, tiny_checkpoint):
    """The composed physics loss, closed over everything but the weights."""
    return make_physics_loss(
        headfield,
        proposal,
        SPHERE_CONFIG,
        SPHERE_CONTAINMENT,
        n_sources=N_SOURCES,
        checkpoint=tiny_checkpoint,
        containment_weight=1.0,
    )


def test_the_composed_loss_is_finite_and_not_degenerate(
    headfield, proposal, tiny_checkpoint, observation, weights
):
    """Gate C, for this composition: a real number in [0, something]."""
    eeg, mask = observation
    value = float(_loss_fn(headfield, proposal, tiny_checkpoint)(weights, eeg, mask))
    assert np.isfinite(value)
    assert 0.0 < value


def test_the_weight_gradient_is_finite_and_non_zero(
    headfield, proposal, tiny_checkpoint, observation, weights
):
    """dL/dweights exists and is not silently zero.

    A zero gradient here is the failure mode that would make every result in this
    package meaningless while looking like it worked: the proposal would be
    trained by the supervised loss alone and the physics term would be decoration.
    """
    eeg, mask = observation
    loss = _loss_fn(headfield, proposal, tiny_checkpoint)
    value, gradient = jax.value_and_grad(loss)(weights, eeg, mask)
    assert np.isfinite(float(value))
    gradient = np.asarray(gradient)
    assert gradient.shape == (weights.shape[0],)
    assert np.isfinite(gradient).all()
    assert np.linalg.norm(gradient) > 0.0


def test_composed_weight_gradient_matches_finite_differences(
    headfield, proposal, tiny_checkpoint, observation, weights
):
    """**The gate.** ``g . d`` against a central difference of the composed forward.

    A directional derivative rather than a coordinate-by-coordinate Jacobian: the
    parameter vector has thousands of entries and every finite-difference
    evaluation costs two full passes through both components, so a random
    direction is what makes the check affordable — and a random direction is a
    strictly harder test than a coordinate, because a bug confined to one
    subsystem cannot hide in a coordinate the sweep skipped.

    The step is chosen where the network's own scale puts the truncation and
    round-off terms in balance; the tolerance is loose enough for that balance and
    tight enough that a wrong chain rule cannot pass.
    """
    eeg, mask = observation
    loss = _loss_fn(headfield, proposal, tiny_checkpoint)
    _, gradient = jax.value_and_grad(loss)(weights, eeg, mask)

    rng = np.random.default_rng(17)
    for trial in range(3):
        direction = rng.normal(size=weights.shape)
        direction /= np.linalg.norm(direction)
        analytic = float(np.asarray(gradient) @ direction)
        step = 1e-4
        high = float(loss(weights + step * direction, eeg, mask))
        low = float(loss(weights - step * direction, eeg, mask))
        numeric = (high - low) / (2.0 * step)
        scale = max(abs(numeric), abs(analytic), 1e-12)
        assert abs(numeric - analytic) / scale < 5e-3, (
            f"direction {trial}: analytic {analytic:.6e} vs numeric {numeric:.6e}"
        )


def test_the_gradient_reaches_the_epoch_as_well(
    headfield, proposal, tiny_checkpoint, observation, weights
):
    """The proposal's VJP differentiates its input too, not only its parameters.

    Not needed for training, and tested anyway: an input gradient that is zero
    while the weight gradient is not would mean the covariance features had been
    detached somewhere, which would be a bug the training loss could not see.
    """
    eeg, mask = observation
    loss = _loss_fn(headfield, proposal, tiny_checkpoint)
    gradient = jax.grad(loss, argnums=1)(weights, eeg, mask)
    assert np.isfinite(np.asarray(gradient)).all()
    assert np.linalg.norm(np.asarray(gradient)) > 0.0


def test_the_position_gradient_is_the_solver_s_and_not_a_shortcut(
    headfield, proposal, tiny_checkpoint, observation, weights
):
    """Freezing the physics component's position sensitivity must change the answer.

    If ``dL/dweights`` were the same with the position derivative switched off,
    the gradient would not be flowing through the position at all — it would be
    reaching the weights by the moment alone, and the OpenMEEG position
    sensitivity would not be load-bearing. This asserts that it is.
    """
    eeg, mask = observation
    loss = _loss_fn(headfield, proposal, tiny_checkpoint)
    reference = np.asarray(jax.grad(loss)(weights, eeg, mask))

    def moment_only(parameters, epoch, channel_mask):
        outputs = proposal_outputs(
            proposal,
            parameters,
            epoch,
            channel_mask,
            n_sources=N_SOURCES,
            checkpoint=tiny_checkpoint,
        )
        frozen = jax.lax.stop_gradient(outputs["positions_m"])
        built = columns(headfield, frozen, outputs["moments"], SPHERE_CONFIG)
        return jnp.mean(projector_residual(built, epoch))

    without = np.asarray(jax.grad(moment_only)(weights, eeg, mask))
    difference = np.linalg.norm(reference - without) / max(
        np.linalg.norm(reference), 1e-30
    )
    assert difference > 1e-3, (
        "blocking the OpenMEEG position sensitivity left the weight gradient "
        f"unchanged to {difference:.2e}; it is not load-bearing"
    )


@pytest.mark.parametrize("count", [1, 2])
def test_the_composition_works_at_every_k(
    headfield, proposal, tiny_checkpoint, observation, weights, count
):
    """K is an input, not a rebuild: the same weights serve every source count."""
    eeg, mask = observation
    loss = make_physics_loss(
        headfield,
        proposal,
        SPHERE_CONFIG,
        SPHERE_CONTAINMENT,
        n_sources=count,
        checkpoint=tiny_checkpoint,
        containment_weight=1.0,
    )
    value, gradient = jax.value_and_grad(loss)(weights, eeg, mask)
    assert np.isfinite(float(value))
    assert np.linalg.norm(np.asarray(gradient)) > 0.0


def test_the_openmeeg_backend_carries_the_same_gradient(headfield):
    """The gate again, on the real BEM and the shipped checkpoint.

    The tests above are about the chain rule, and the sphere backend serves that
    perfectly well. This one is about the *claim*: that the gradient which trains
    the network has crossed a compiled C++ boundary-element solver. It runs the
    full-sized packaged network against the cached fsaverage operator and checks
    one directional derivative — the same measurement
    ``scripts/report_hybrid_gradcheck.py`` writes into
    ``results/hybrid/gradcheck.json``, at one direction instead of three so it
    belongs in a test suite.

    Skipped when the head-model artifact or the trained checkpoint is absent, both
    of which are build products a clean checkout does not necessarily have.
    """
    from neurolayout.clients import COMPONENT_ROOT
    from neurolayout.hybrid.model import flat_parameters, load_checkpoint

    artifact = default_artifact_path()
    checkpoint = COMPONENT_ROOT / "proposal" / "proposal.pt"
    if not artifact.exists():
        pytest.skip("no OpenMEEG head-model artifact")
    if not checkpoint.exists():
        pytest.skip("no trained proposal checkpoint; run scripts/train_proposal.py")

    geometry = HeadGeometry.load(artifact)
    config = LocalizeConfig(backend="openmeeg", n_times=16, sfreq=160.0)
    containment = Containment.from_points(geometry.vertices[0])
    model, _ = load_checkpoint(checkpoint)
    weights = jnp.asarray(flat_parameters(model).numpy())

    rng = np.random.default_rng(21)
    eeg = rng.normal(0.0, 1e-6, (1, geometry.n_channels, 16))
    eeg -= eeg.mean(axis=1, keepdims=True)
    epoch, mask = jnp.asarray(eeg), jnp.ones((1, geometry.n_channels))

    with open_component("proposal", "local") as proposal:
        loss = make_physics_loss(
            headfield, proposal, config, containment, n_sources=2
        )
        value, gradient = jax.value_and_grad(loss)(weights, epoch, mask)
        assert np.isfinite(float(value))
        gradient = np.asarray(gradient)
        assert gradient.shape == (weights.shape[0],)
        assert np.linalg.norm(gradient) > 0.0

        direction = rng.normal(size=gradient.shape)
        direction /= np.linalg.norm(direction)
        analytic = float(gradient @ direction)
        step = 1e-4
        numeric = (
            float(loss(weights + step * direction, epoch, mask))
            - float(loss(weights - step * direction, epoch, mask))
        ) / (2.0 * step)
    scale = max(abs(numeric), abs(analytic), 1e-12)
    assert abs(numeric - analytic) / scale < 5e-3, (
        f"through the real BEM: analytic {analytic:.6e} vs numeric {numeric:.6e}"
    )
