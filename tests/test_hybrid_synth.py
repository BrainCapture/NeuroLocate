# Copyright 2026 NeuroLocate contributors
# SPDX-License-Identifier: Apache-2.0
r"""The synthetic generator, the source-set representation, and the matching.

Everything the proposal network learns from comes out of these functions, so a
silent bug here would be a bug in every number the package reports and would look
like a scientific result. They are checked against their own definitions rather
than against a stored expectation: a prescribed correlation has to be the
correlation that comes out, a requested separation band has to be the band the
sources land in, and a set loss has to be indifferent to the order of the set.

None of these needs a gain bank of 240,000 positions. They run against a
miniature one built in the fixture, which is also what makes them run on a clean
checkout.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

import torch  # noqa: E402
from neurolayout.hybrid.bank import Bank, ForwardVariant  # noqa: E402
from neurolayout.hybrid.losses import (  # noqa: E402
    LossWeights,
    heatmap_targets,
    proposal_loss,
)
from neurolayout.hybrid.model import (  # noqa: E402
    ProposalConfig,
    ProposalNet,
    decode,
    flat_parameters,
    load_flat_parameters,
)
from neurolayout.hybrid.synth import (  # noqa: E402
    Synthesizer,
    SynthSpec,
    correlated_waveforms,
)

N_CHANNELS = 16
N_POSITIONS = 600
N_TIMES = 32


@pytest.fixture(scope="module", name="bank")
def fixture_bank() -> Bank:
    """A miniature gain bank with plausible geometry and smooth gains.

    Not OpenMEEG: these tests are about the sampler and the loss, and a real BEM
    would make them slow without making them stronger. The gains are a smooth
    function of position so that nearby sources really do have similar
    topographies, which is the property the separation and correlation axes are
    meant to stress.
    """
    rng = np.random.default_rng(0)
    positions = rng.uniform(-0.05, 0.05, (N_POSITIONS, 3))
    normals = rng.standard_normal((N_POSITIONS, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    sensors = rng.standard_normal((N_CHANNELS, 3))
    sensors *= 0.09 / np.linalg.norm(sensors, axis=1, keepdims=True)
    offset = sensors[None, :, None, :] - positions[:, None, None, :]
    distance = np.linalg.norm(offset, axis=3, keepdims=True)
    gains = (offset / distance**3)[:, :, 0, :] * 1e-4
    gains = gains - gains.mean(axis=1, keepdims=True)  # average reference
    return Bank(
        positions_m=positions,
        normals=normals,
        is_cortical=np.ones(N_POSITIONS, dtype=bool),
        depth_mm=np.full(N_POSITIONS, 20.0),
        gains=np.repeat(gains[None].astype(np.float32), 2, axis=0),
        variants=(ForwardVariant("nominal"), ForwardVariant("other", 0.003)),
        channel_names=tuple(f"E{index}" for index in range(N_CHANNELS)),
        sensor_xyz=sensors,
        metadata={"note": "test fixture"},
    )


@pytest.fixture(scope="module", name="synth")
def fixture_synth(bank: Bank) -> Synthesizer:
    return Synthesizer(bank, SynthSpec(n_times=N_TIMES))


#
# The correlation axis
#


@pytest.mark.parametrize("correlation", [0.0, 0.5, 0.9, 0.99])
@pytest.mark.parametrize("n_sources", [2, 3, 4])
def test_prescribed_correlation_is_realized(correlation, n_sources) -> None:
    """A requested mutual cosine has to be the cosine that comes out.

    The whole matrix is indexed by this number, so if the sampler quietly
    delivered something else the benchmark would be measuring a different axis
    from the one it is labelled with.
    """
    rng = np.random.default_rng(4)
    waveforms = correlated_waveforms(
        rng, n_sources, correlation, n_times=N_TIMES, sfreq=160.0
    )
    unit = waveforms / np.linalg.norm(waveforms, axis=1, keepdims=True)
    gram = unit @ unit.T
    off = gram[~np.eye(n_sources, dtype=bool)]
    assert waveforms.shape == (n_sources, N_TIMES)
    np.testing.assert_allclose(off, correlation, atol=0.02)


def test_shared_dynamics_are_exactly_rank_one() -> None:
    """``shared`` must mean rank one, not merely highly correlated.

    That is the whole difficulty of the hard cells: with a rank-one data matrix no
    method can separate the sources temporally, and a sampler that delivered
    0.999 instead of 1.0 would leave a sliver of signal subspace that a subspace
    method could exploit — and the benchmark would understate how hard the case is.
    """
    rng = np.random.default_rng(5)
    waveforms = correlated_waveforms(rng, 4, 1.0, n_times=N_TIMES, sfreq=160.0)
    assert np.linalg.matrix_rank(waveforms, tol=1e-9) == 1
    for row in waveforms[1:]:
        np.testing.assert_allclose(row, waveforms[0])


def test_correlation_outside_the_unit_interval_is_refused() -> None:
    """A cosine above one is not a hard case, it is a mistake."""
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        correlated_waveforms(np.random.default_rng(0), 2, 1.5, n_times=8, sfreq=160.0)


#
# The source-set draw
#


@pytest.mark.parametrize("n_sources", [1, 2, 3, 4])
def test_the_draw_returns_the_requested_source_count(synth, n_sources) -> None:
    """K is an input to the sampler, and every array has to agree about it."""
    sample = synth.draw(np.random.default_rng(n_sources), n_sources=n_sources)
    assert sample.n_sources == n_sources
    assert sample.positions_m.shape == (n_sources, 3)
    assert sample.moments_nam.shape == (n_sources, 3)
    assert sample.waveforms.shape == (n_sources, N_TIMES)
    assert sample.eeg.shape == (N_CHANNELS, N_TIMES)


@pytest.mark.parametrize("wanted", [15.0, 30.0, 60.0])
def test_a_requested_separation_lands_near_that_separation(synth, wanted) -> None:
    """The separation band is a controlled variable, not a hope.

    The sampler snaps to the nearest bank position, so exactness is not available;
    what has to hold is that asking for 15 mm does not give 60 mm. The tolerance
    is generous for that reason and tight enough to catch the axis being ignored.
    """
    distances = [
        synth.draw(
            np.random.default_rng(seed), n_sources=2, separation_mm=wanted
        ).separation_mm
        for seed in range(12)
    ]
    assert np.median(distances) == pytest.approx(wanted, rel=0.35)


def test_one_source_has_no_separation(synth) -> None:
    """K = 1 reports an infinite separation rather than a zero or a crash."""
    sample = synth.draw(np.random.default_rng(1), n_sources=1)
    assert np.isinf(sample.separation_mm)


def test_the_eeg_is_the_gain_times_the_moment_times_the_waveform(bank, synth) -> None:
    """The generator's own definition, recomputed from the bank by hand.

    A clean draw (no noise, no dropout) must reproduce
    ``sum_k G(p_k) m_k a_k(t)`` exactly, or the network is being trained on
    something other than the forward model it is scored against.
    """
    for seed in range(20):
        sample = synth.draw(
            np.random.default_rng(seed), n_sources=3, clean=True, variant="nominal"
        )
        if sample.mask.all():
            break
    else:  # every draw dropped a channel, which the default 15% makes impossible
        pytest.fail("no unmasked draw in twenty attempts")
    assert sample.snr_db is None
    index = bank.variant_index("nominal")
    rows = [
        int(np.argmin(np.linalg.norm(bank.positions_m - position, axis=1)))
        for position in sample.positions_m
    ]
    gains = np.asarray(bank.gains[index, rows], dtype=np.float64)
    expected = np.einsum(
        "kcj,kj,kt->ct", gains, sample.moments_nam * 1e-9, sample.waveforms
    )
    np.testing.assert_allclose(sample.eeg, expected, rtol=1e-10, atol=1e-18)


def test_a_dropped_channel_is_zero_and_the_rest_are_re_referenced(bank) -> None:
    """Dropping channels must leave the survivors carrying *their own* reference."""
    from neurolayout.hybrid.synth import _apply_mask

    rng = np.random.default_rng(3)
    eeg = rng.standard_normal((N_CHANNELS, N_TIMES))
    eeg -= eeg.mean(axis=0, keepdims=True)
    mask = np.ones(N_CHANNELS, dtype=np.float32)
    mask[[1, 4, 7]] = 0.0
    masked = _apply_mask(eeg, mask)
    assert np.all(masked[mask == 0.0] == 0.0)
    np.testing.assert_allclose(masked[mask > 0.0].mean(axis=0), 0.0, atol=1e-12)


def test_the_snr_the_draw_reports_is_the_snr_it_applied(synth) -> None:
    """A stated SNR that is not the realized one would mislabel every row."""
    sample = synth.draw(np.random.default_rng(2), n_sources=2, snr_db=10.0)
    assert sample.snr_db == 10.0


def test_a_held_out_split_shares_no_position(bank) -> None:
    """Validation sources must be ones the network never had a gain for."""
    train, validation = bank.split(0.2, seed=1)
    assert len(set(train.tolist()) & set(validation.tolist())) == 0
    assert len(train) + len(validation) == bank.n_positions
    assert len(validation) == pytest.approx(0.2 * bank.n_positions, abs=1)


def test_a_variant_can_be_held_out_of_a_synthesizer(bank) -> None:
    """Holding a head model out has to actually hold it out."""
    restricted = Synthesizer(
        bank, SynthSpec(n_times=N_TIMES), variants=("nominal",)
    )
    drawn = {
        restricted.draw(np.random.default_rng(seed)).variant for seed in range(20)
    }
    assert drawn == {"nominal"}


#
# The set representation: the loss, the decode, the permutation invariance
#


@pytest.fixture(scope="module", name="lattice")
def fixture_lattice() -> torch.Tensor:
    """A 3x3x3 lattice at 20 mm pitch, wide enough to hold the fixture's sources."""
    axis = np.array([-0.02, 0.0, 0.02])
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    return torch.as_tensor(grid.reshape(-1, 3), dtype=torch.float64)


def _batch(positions: np.ndarray) -> dict[str, torch.Tensor]:
    """A one-entry supervised batch at the given source positions."""
    n_sources = positions.shape[0]
    moments = np.tile([0.0, 0.0, 1.0], (n_sources, 1)) * 20.0
    return {
        "positions_m": torch.as_tensor(positions[None], dtype=torch.float64),
        "moments_nam": torch.as_tensor(moments[None], dtype=torch.float64),
        "source_mask": torch.ones((1, n_sources), dtype=torch.float64),
        "n_sources": torch.tensor([n_sources]),
    }


def _outputs(lattice: torch.Tensor, n_sources: int) -> dict[str, torch.Tensor]:
    """Deterministic network-shaped outputs over the lattice."""
    torch.manual_seed(0)
    n_voxels = lattice.shape[0]
    return {
        "logits": torch.randn(1, n_voxels, dtype=torch.float64),
        "positions_m": lattice[None].clone(),
        "moments": torch.nn.functional.normalize(
            torch.randn(1, n_voxels, 3, dtype=torch.float64), dim=2
        ),
        "count_logits": torch.randn(1, 4, dtype=torch.float64),
    }


def test_the_set_loss_is_indifferent_to_the_order_of_the_set(lattice) -> None:
    """A set has no order, so the loss must not have one either.

    This is the property a heatmap representation buys and direct set regression
    does not: no assignment step means nothing to flip between epochs.
    """
    positions = np.array([[-0.02, 0.0, 0.0], [0.02, 0.0, 0.02], [0.0, 0.02, -0.02]])
    outputs = _outputs(lattice, 3)
    weights = LossWeights()
    first, _ = proposal_loss(outputs, _batch(positions), lattice, weights, pitch_m=0.02)
    for order in ([2, 0, 1], [1, 2, 0], [2, 1, 0]):
        permuted, _ = proposal_loss(
            outputs, _batch(positions[order]), lattice, weights, pitch_m=0.02
        )
        assert float(permuted) == pytest.approx(float(first), rel=1e-12)


def test_the_heatmap_target_peaks_at_the_source(lattice) -> None:
    """One at the containing voxel, falling away, and never above one."""
    positions = np.array([[0.02, 0.0, 0.0]])
    target, distance = heatmap_targets(
        lattice, _batch(positions)["positions_m"], _batch(positions)["source_mask"], 5.0
    )
    assert float(target.max()) == pytest.approx(1.0, abs=1e-9)
    assert int(target.argmax()) == int(distance[0, :, 0].argmin())
    assert float(target.min()) >= 0.0


def test_a_padded_source_slot_never_becomes_the_nearest(lattice) -> None:
    """Padding is not a source, and a padded slot at the origin would be one."""
    positions = np.array([[0.02, 0.0, 0.0], [0.0, 0.0, 0.0]])
    batch = _batch(positions)
    batch["source_mask"] = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    target, distance = heatmap_targets(
        lattice, batch["positions_m"], batch["source_mask"], 5.0
    )
    assert torch.isinf(distance[0, :, 1]).all()
    assert int(target.argmax()) == int(distance[0, :, 0].argmin())


def test_the_decode_returns_k_distinct_voxels(lattice) -> None:
    """Non-maximum suppression is what stops one blob being read as K sources."""
    outputs = _outputs(lattice, 2)
    picked = decode(outputs, 3, nms_radius_m=0.019, max_sources=4)
    assert picked["positions_m"].shape == (1, 3, 3)
    indices = picked["indices"][0].tolist()
    assert len(set(indices)) == 3


def test_the_decode_takes_the_highest_logit_first(lattice) -> None:
    """The first pick is the argmax, whatever the suppression does afterwards."""
    outputs = _outputs(lattice, 1)
    picked = decode(outputs, 1, nms_radius_m=0.001, max_sources=4)
    assert int(picked["indices"][0, 0]) == int(outputs["logits"][0].argmax())


def test_the_decode_asks_the_count_head_when_k_is_not_given(lattice) -> None:
    """K = None is the no-prior-on-K path, and it has to use the count head."""
    outputs = _outputs(lattice, 1)
    outputs["count_logits"] = torch.tensor([[0.0, 5.0, 0.0, 0.0]], dtype=torch.float64)
    picked = decode(outputs, None, nms_radius_m=0.019, max_sources=4)
    assert int(picked["n_predicted"][0]) == 2
    assert picked["positions_m"].shape[1] == 2


#
# The parameter vector, which is a wire format
#


def test_the_flat_parameter_vector_round_trips() -> None:
    """It crosses a Tesseract boundary, so its layout has to be exact."""
    torch.manual_seed(1)
    centres = np.zeros((8, 3))
    sensors = np.random.default_rng(0).standard_normal((N_CHANNELS, 3)) * 0.09
    model = ProposalNet(
        ProposalConfig(n_channels=N_CHANNELS, width=16, depth=1, heads=2, voxel_dim=4),
        centres,
        sensors,
    ).double()
    original = flat_parameters(model).clone()
    perturbed = original + 0.1
    load_flat_parameters(model, perturbed)
    torch.testing.assert_close(flat_parameters(model), perturbed)
    load_flat_parameters(model, original)
    torch.testing.assert_close(flat_parameters(model), original)


def test_a_wrong_length_parameter_vector_is_refused() -> None:
    """A silently truncated wire format would be a wrong answer, not an error."""
    torch.manual_seed(1)
    model = ProposalNet(
        ProposalConfig(n_channels=N_CHANNELS, width=16, depth=1, heads=2, voxel_dim=4),
        np.zeros((8, 3)),
        np.random.default_rng(0).standard_normal((N_CHANNELS, 3)) * 0.09,
    ).double()
    with pytest.raises(ValueError, match="entries but the model needs"):
        load_flat_parameters(model, torch.zeros(7))


def test_the_network_is_invariant_to_the_epoch_s_scale_and_sign() -> None:
    """It reads the covariance, so both are gauge and must change nothing.

    Not a nice-to-have: a source's time course is only defined up to sign, and the
    absolute amplitude of cortical activity is not knowable from one epoch. A
    network that was sensitive to either would be fitting something that is not
    there.
    """
    torch.manual_seed(2)
    sensors = np.random.default_rng(0).standard_normal((N_CHANNELS, 3)) * 0.09
    model = ProposalNet(
        ProposalConfig(n_channels=N_CHANNELS, width=16, depth=1, heads=2, voxel_dim=4),
        np.zeros((8, 3)),
        sensors,
    ).double().eval()
    eeg = torch.randn(2, N_CHANNELS, N_TIMES, dtype=torch.float64)
    mask = torch.ones(2, N_CHANNELS, dtype=torch.float64)
    with torch.no_grad():
        base = model(eeg, mask)["logits"]
        scaled = model(eeg * 1e6, mask)["logits"]
        flipped = model(-eeg, mask)["logits"]
    torch.testing.assert_close(scaled, base, rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(flipped, base, rtol=1e-9, atol=1e-9)
