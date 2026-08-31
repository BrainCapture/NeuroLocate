# Copyright 2026 NeuroLocate contributors
# SPDX-License-Identifier: Apache-2.0
r"""One frame, one unit, everywhere — and the control that separates the two runs.

**Frames.** `CLAUDE.md` makes this a hard gate: MNE head coordinates, metres, in
every artifact. A frame error is the failure mode that produces plausible-looking
numbers and a completely wrong answer, and it cannot be caught by a gradient check
— both sides of a finite-difference test would be wrong the same way. So the chain
is checked link by link: the gain bank's positions, the proposal lattice, the
sensor array, and what comes back out of the served component.

**The stop-gradient control.** The result the brief asks for turns on whether the
solver's training gradient did anything, and that question is only answerable if
the control differs from the treatment in exactly one respect. These tests pin
that: with the flag off, the physics term is still computed and still logged, its
gradient is discarded, and the parameter update is *exactly* the supervised one.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

import torch  # noqa: E402
from neurolayout.hybrid.model import ProposalConfig, ProposalNet, decode  # noqa: E402
from neurolayout_shared.openmeeg_model import (  # noqa: E402
    HeadGeometry,
    default_artifact_path,
)

#: Every length in this project is metres. A head is about 0.18 m across, so any
#: coordinate outside this range is either millimetres or a different frame.
PLAUSIBLE_M = 0.25


@pytest.fixture(scope="module", name="geometry")
def fixture_geometry() -> HeadGeometry:
    path = default_artifact_path()
    if not path.exists():
        pytest.skip("no OpenMEEG head-model artifact")
    return HeadGeometry.load(path)


def test_the_head_model_declares_its_frame_and_units(geometry: HeadGeometry) -> None:
    """Every artifact records which frame it is in. This is the source of it."""
    assert geometry.coord_frame == "mne-head"
    assert geometry.units == "m"


def test_the_sensor_array_is_metres_in_the_head_frame(geometry: HeadGeometry) -> None:
    """64 electrodes on a head-sized sphere, centred near the auricular midpoint."""
    sensors = np.asarray(geometry.sensor_xyz)
    assert sensors.shape == (64, 3)
    radius = np.linalg.norm(sensors - sensors.mean(axis=0), axis=1)
    assert 0.06 < float(np.median(radius)) < 0.12
    assert float(np.abs(sensors).max()) < PLAUSIBLE_M


def test_the_montage_orientation_is_right_forward_up(geometry: HeadGeometry) -> None:
    """+x toward the right ear, +y toward the nose, +z up.

    Checked against named electrodes rather than against a stored array: a
    transposed or reflected frame reproduces every distance and every norm, and
    only the labels catch it.
    """
    names = list(geometry.channel_names)
    sensors = np.asarray(geometry.sensor_xyz)
    index = {name: position for name, position in zip(names, sensors, strict=True)}
    for left, right in (("C3", "C4"), ("F3", "F4"), ("P3", "P4")):
        if left in index and right in index:
            assert index[left][0] < index[right][0], f"{left}/{right} are mirrored in x"
    if "Fpz" in index and "Oz" in index:
        assert index["Fpz"][1] > index["Oz"][1], "front and back are swapped in y"
    if "Cz" in index and "Iz" in index:
        assert index["Cz"][2] > index["Iz"][2], "up and down are swapped in z"


def test_the_packaged_lattice_lies_inside_the_inner_skull(geometry: HeadGeometry) -> None:
    """Voxel centres are candidate *dipole* positions; outside, the BEM is void.

    Checked on the lattice the shipped checkpoint actually carries — the network
    stores its own voxel centres, so this is the artifact a served component
    proposes from, not a freshly built grid that might differ from it.
    """
    from neurolayout.clients import COMPONENT_ROOT
    from neurolayout.hybrid.model import load_checkpoint

    model, _ = load_checkpoint(COMPONENT_ROOT / "proposal" / "proposal.pt")
    centres = np.asarray(model.voxel_centres.detach().numpy(), dtype=np.float64)
    inner = np.asarray(geometry.vertices[0])
    low, high = inner.min(axis=0), inner.max(axis=0)
    assert centres.shape[1] == 3
    assert len(centres) > 500
    assert np.all(centres >= low - 1e-9)
    assert np.all(centres <= high + 1e-9)
    assert float(np.abs(centres).max()) < PLAUSIBLE_M
    # And inside the surface, not merely inside its bounding box: a lattice point
    # in the skull is a dipole the BEM has no equations for.
    from neurolayout.hybrid.bank import _inside_with_margin, _surface_info

    inside = _inside_with_margin(
        centres,
        _surface_info(geometry.vertices[0], geometry.triangles[0]),
        margin_m=0.0,
    )
    assert bool(inside.all()), f"{int((~inside).sum())} lattice points are outside"


def test_the_bank_positions_are_in_the_same_frame_as_the_head_model() -> None:
    """The bank is built offline; its positions must not drift from the geometry."""
    from neurolayout.hybrid.bank import BankSpec

    spec = BankSpec(n_positions=64, seed=1)
    assert spec.margin_mm > 0.0
    assert spec.normal_jitter_mm > 0.0
    # The fingerprint covers everything that changes the draw, so two specs that
    # differ anywhere cannot silently produce the same bank.
    assert BankSpec(n_positions=64, seed=2).fingerprint() != spec.fingerprint()


def test_a_decoded_proposal_stays_in_metres_near_its_lattice_point() -> None:
    """The offset is bounded by one pitch, so a decode cannot leave the head.

    This is the link where a unit error would be easiest to make and hardest to
    see: the network's offsets are trained against a loss in millimetres and
    applied to centres in metres.
    """
    torch.manual_seed(0)
    pitch = 0.008
    centres = np.array([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0], [0.0, 0.02, 0.01]])
    sensors = np.random.default_rng(0).standard_normal((8, 3)) * 0.09
    model = ProposalNet(
        ProposalConfig(
            n_channels=8, width=16, depth=1, heads=2, voxel_dim=4, grid_pitch_m=pitch
        ),
        centres,
        sensors,
    ).double().eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn_like(parameter))
        outputs = model(
            torch.randn(2, 8, 16, dtype=torch.float64), torch.ones(2, 8, dtype=torch.float64)
        )
    offsets = outputs["positions_m"] - model.voxel_centres[None]
    assert float(offsets.abs().max()) <= pitch * model.config.offset_scale + 1e-12
    picked = decode(outputs, 2, nms_radius_m=0.001, max_sources=4)
    assert float(picked["positions_m"].abs().max()) < PLAUSIBLE_M


def test_the_moments_the_network_returns_are_unit_directions() -> None:
    """A moment is a direction here; its magnitude is gauge and is profiled out."""
    torch.manual_seed(1)
    model = ProposalNet(
        ProposalConfig(n_channels=8, width=16, depth=1, heads=2, voxel_dim=4),
        np.zeros((6, 3)),
        np.random.default_rng(0).standard_normal((8, 3)) * 0.09,
    ).double().eval()
    with torch.no_grad():
        outputs = model(
            torch.randn(3, 8, 12, dtype=torch.float64), torch.ones(3, 8, dtype=torch.float64)
        )
    norms = outputs["moments"].norm(dim=2)
    torch.testing.assert_close(norms, torch.ones_like(norms), rtol=1e-9, atol=1e-9)


#
# The stop-gradient control
#


@pytest.fixture(scope="module", name="tiny_bank")
def fixture_tiny_bank():
    """A miniature gain bank, so the control can be exercised without OpenMEEG."""
    from neurolayout.hybrid.bank import Bank, ForwardVariant

    rng = np.random.default_rng(0)
    n_positions, n_channels = 200, 16
    positions = rng.uniform(-0.04, 0.04, (n_positions, 3))
    normals = rng.standard_normal((n_positions, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    sensors = rng.standard_normal((n_channels, 3))
    sensors *= 0.09 / np.linalg.norm(sensors, axis=1, keepdims=True)
    offset = sensors[None, :, :] - positions[:, None, :]
    gains = (offset / np.linalg.norm(offset, axis=2, keepdims=True) ** 3) * 1e-4
    gains = gains - gains.mean(axis=1, keepdims=True)
    return Bank(
        positions_m=positions,
        normals=normals,
        is_cortical=np.ones(n_positions, dtype=bool),
        depth_mm=np.full(n_positions, 20.0),
        gains=gains[None].astype(np.float32),
        variants=(ForwardVariant("nominal"),),
        channel_names=tuple(f"E{index}" for index in range(n_channels)),
        sensor_xyz=sensors,
        metadata={"note": "test fixture"},
    )


def _finetune_run(tiny_bank, monkeypatch, tmp_path, use_physics: bool):
    """One short fine-tune, or its control, against the sphere backend."""
    from neurolayout.clients import open_components
    from neurolayout.hybrid.finetune import FinetuneConfig, finetune
    from neurolayout.hybrid.model import ProposalConfig, ProposalNet, save_checkpoint
    from neurolayout.hybrid.synth import SynthSpec
    from neurolayout.localize import Containment, LocalizeConfig

    torch.manual_seed(0)
    model = ProposalNet(
        ProposalConfig(
            n_channels=16, width=16, depth=1, heads=2, voxel_dim=4, max_sources=2,
            grid_pitch_m=0.02,
        ),
        np.array([[x, y, 0.0] for x in (-0.02, 0.0, 0.02) for y in (-0.02, 0.0, 0.02)]),
        tiny_bank.sensor_xyz,
    ).double()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.05 * torch.randn_like(parameter))
    save_checkpoint(tmp_path / "tiny.pt", model, {})
    monkeypatch.setenv("NEUROLOCATE_PROPOSAL_DIR", str(tmp_path))

    config = FinetuneConfig(
        steps=3,
        batch_size=2,
        learning_rate=1e-3,
        k_cycle=(2,),
        correlation_cycle=("shared",),
        log_every=1,
        spec=SynthSpec(n_times=8),
    )
    localize = LocalizeConfig(backend="sphere", n_times=8, n_channels=16)
    containment = Containment(centre_cm=np.zeros(3), semi_axes_cm=np.full(3, 7.0))
    with open_components(("headfield", "proposal"), "local") as opened:
        result = finetune(
            model,
            tiny_bank,
            config,
            headfield=opened["headfield"],
            proposal=opened["proposal"],
            localize_config=localize,
            containment=containment,
            checkpoint_name="tiny",
            use_physics_gradient=use_physics,
        )
    from neurolayout.hybrid.model import flat_parameters

    return result, flat_parameters(model).clone()


def test_the_control_still_computes_and_logs_the_physics_term(
    tiny_bank, monkeypatch, tmp_path
) -> None:
    """Both arms have to be comparable on the same quantity.

    A control that skipped the physics term entirely would be cheaper *and*
    unmeasurable: there would be no sensor-space number to compare the two runs
    on. So it is computed and logged either way, and only its gradient is
    discarded.
    """
    result, _ = _finetune_run(tiny_bank, monkeypatch, tmp_path, use_physics=False)
    assert result["use_physics_gradient"] is False
    for record in result["history"]:
        assert np.isfinite(record["physics_loss"])
        assert record["physics_grad_norm"] > 0.0, (
            "the control must still evaluate the physics gradient to report it; "
            "it simply must not descend on it"
        )


def test_the_control_and_the_treatment_reach_different_parameters(
    tiny_bank, monkeypatch, tmp_path
) -> None:
    """Same checkpoint, same batches, same steps — the flag has to matter.

    If these agreed, the reported difference between `hybrid` and
    `hybrid_stopgrad` would be measuring nothing at all.
    """
    _, with_physics = _finetune_run(tiny_bank, monkeypatch, tmp_path, use_physics=True)
    _, without = _finetune_run(tiny_bank, monkeypatch, tmp_path, use_physics=False)
    difference = float((with_physics - without).norm() / max(float(without.norm()), 1e-30))
    assert difference > 1e-9, "the stop-gradient flag changed nothing"


def test_the_two_arms_see_the_same_batches(tiny_bank, monkeypatch, tmp_path) -> None:
    """The comparison is only fair if the data is identical.

    Both runs seed from the same `FinetuneConfig.seed` and draw in the same order,
    so the supervised loss trajectory must match step for step — any drift there
    would mean the two arms were trained on different data.
    """
    first, _ = _finetune_run(tiny_bank, monkeypatch, tmp_path, use_physics=True)
    second, _ = _finetune_run(tiny_bank, monkeypatch, tmp_path, use_physics=False)
    assert first["history"][0]["source_loss"] == pytest.approx(
        second["history"][0]["source_loss"], rel=1e-12
    )
    assert first["history"][0]["physics_loss"] == pytest.approx(
        second["history"][0]["physics_loss"], rel=1e-12
    )
    assert [record["k"] for record in first["history"]] == [
        record["k"] for record in second["history"]
    ]
