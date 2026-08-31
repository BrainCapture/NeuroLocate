# The frozen benchmark

This records what was measured, on what, and how, for every number the README
prints. It is a record of one frozen synthetic experiment, not a study.

Regenerate every table below from the committed shards with `make summary`.

## Identity

| | |
| --- | --- |
| matrix fingerprint | `35b6e07a7e1307313d3a9b13c62f3225a8e6f68ddc0e938b313265b45986b342` |
| observation-artifact fingerprint | `b08fbe96a062291d26cf3fbf84f9347e45799a63919417f04f61937a61e993e4` |
| head-model fingerprint (inference) | `1d03f251f3d0e9f37c3d1a75810800eca17282a1203ca05de444237d6cfdc7c7` |
| conditions | 10 |
| trials | 80 (8 per condition) |
| methods | 7 |
| OpenMEEG | 2.5.16 |
| MNE-Python (generator only) | 1.12.1 |

The matrix fingerprint covers the condition definitions, the method list and the
restart count. It was declared and committed — `results/hybrid_freeze.json` —
before any result existed. The benchmark runner refuses to start against
observations built for a different matrix, and `make summary` refuses shards
produced for one. `tests/test_hybrid_provenance.py` checks that the committed
freeze still matches what the source declares.

`results/hybrid_freeze.json` is reproduced here byte-for-byte from the repository
the benchmark was run in — that is what makes it evidence rather than a
restatement — so its `frozen_at_commit` and its prose note refer to that
repository's commit and filenames, not to this one. Only its `fingerprint` field
is load-bearing, and that is the field the test checks. It also records a
superseded declaration and why it was replaced: the four-restart arm and the
closed-form moment warm start were both added to strengthen the baselines, while
no result yet existed to choose them from.

## What K means

`K` is the number of simultaneously active compact sources an estimator is asked
to find. The cortical source space holds about 20,000 candidate locations and
there are 64 sensors, so the unrestricted problem is under-determined and has no
unique solution. Assuming a small, known `K` — the same assumption classical
equivalent-current-dipole fitting makes — is what makes a recovered position
interpretable.

**Every method in this matrix is given `K`**, including every baseline, and every
method returns exactly `K` estimates. The proposal network also predicts a count,
but the matrix does not run on it.

## The conditions

| condition | K | dynamics | true separation | SNR |
| --- | --- | --- | --- | --- |
| `h-k1` | 1 | — | — | 20 dB |
| `h-k2-distinct` | 2 | independent time courses | 28–40 mm | 20 dB |
| `h-k2-correlated` | 2 | correlation 0.9 | 28–40 mm | 20 dB |
| `h-k2-shared` | 2 | one shared time course | 28–40 mm | 20 dB |
| `h-k2-shared-close` | 2 | one shared time course | 12–18 mm | 20 dB |
| `h-k2-shared-10db` | 2 | one shared time course | 28–40 mm | 10 dB |
| `h-k4-distinct` | 4 | independent time courses | 30–140 mm | 20 dB |
| `h-k4-correlated` | 4 | correlation 0.9 | 30–140 mm | 20 dB |
| `h-k4-shared` | 4 | one shared time course | 30–140 mm | 20 dB |
| `h-k4-shared-10db` | 4 | one shared time course | 30–140 mm | 10 dB |

The headline trial and the demo are in `h-k2-shared-close`: K=2, one shared time
course, 20 dB SNR, sources 15.2 mm apart. When sources share a time course the
sensor data matrix is rank one, which is the regime this matrix exists to probe.

Epoch: 32 samples at 160 Hz, 64 channels, average-referenced, in one frozen
channel order. Truths are placed deliberately between ico5 candidate positions,
so no dictionary method can be exactly right.

## Observation and inference use different forward models

| | generator | inference |
| --- | --- | --- |
| solver | MNE-Python linear-collocation BEM | OpenMEEG symmetric BEM |
| surfaces | fsaverage ico4 | fsaverage ico3 |
| skull conductivity | 0.0033 S/m (brain:skull 91:1) | 0.006 S/m (50:1) |
| electrodes | displaced, 5.09 mm realized RMS | nominal montage |

There is **no matched condition** in this matrix. No estimator has the model that
made the data, and none of the six head models the proposal network trained on is
the generator either.

## Recovery

A source counts as **recovered** when the estimate it was matched with lands
within 20 mm of it. Matching is the assignment that minimizes the total distance
over all `K` estimates. Because every method is handed `K` and returns exactly
`K` estimates, the assignment is a bijection: a missed source is simultaneously a
false positive, so recall and precision are the same number and are reported once.

Per-source recovery at K=2, over 80 sources in 40 trials:

| method | sources recovered | median worst-source mm | median s / trial (whole matrix) |
| --- | --- | --- | --- |
| uninformed initialization + refinement | 41.3% | 26.3 | 19.2 |
| RAP-MUSIC | 57.5% | 40.0 | 6.5 |
| OpenMEEG dipole scan | 61.3% | 23.7 | 2.5 |
| four physical restarts | 71.3% | 20.9 | 69.0 |
| learned proposal alone | 80.0% | 18.4 | 0.0 |
| proposal + physical refinement | 82.5% | 16.3 | 16.8 |

Each per-trial error is the **worst** of that trial's sources, because a trial
that recovers one source and misses the other is a failed trial.

**Most of the recovery gain is in the initialization.** 41.3% → 80.0% is what
changing the starting point buys. 80.0% → 82.5% is what the physical refinement
adds on top of it. The refinement's contribution is clearer in continuous
localization error — the paired table below — than in the recovery rate.

![Recovery and inference time, by method](figures/benchmark.png)

*Recomputed from the shards by `scripts/plot_benchmark.py` every time it runs.*

## Paired comparisons

Per-trial paired differences on the correlated and shared-dynamics cells, 56
trials. Negative means the composition is closer to the truth. Intervals are a
4,000-draw percentile bootstrap over trials.

| vs | n | median difference mm | 95% CI | excludes zero |
| --- | --- | --- | --- | --- |
| learned proposal alone | 56 | -4.1 | -7.5 to -2.7 | yes |
| uninformed initialization + refinement | 56 | -6.7 | -12.9 to -2.8 | yes |
| RAP-MUSIC | 56 | -16.7 | -26.4 to -3.4 | yes |
| OpenMEEG dipole scan | 56 | +1.8 | -3.1 to +7.0 | no |
| four physical restarts | 56 | +0.3 | -4.2 to +4.7 | no |

## The K=4 result

Split by source count, the same paired difference. Negative means the
composition is closer:

| vs | K=1 | K=2 | K=4 |
| --- | --- | --- | --- |
| uninformed initialization + refinement | -0.0 | -13.5 | +4.2 |
| RAP-MUSIC | +0.6 | -19.0 | **+29.9** |
| four physical restarts | -0.0 | -2.4 | +24.3 |
| learned proposal alone | +0.9 | -3.2 | -13.3 |

**K=4 is not solved by the current proposal.** RAP-MUSIC is 29.9 mm closer than
the composition in the frozen K=4 comparison. The refinement still improves the
proposal at K=4 (-13.3 mm), so what fails at K=4 is the proposal, not the
physics. The open circles in the figure above are the same recovery rate at
K=4: every method's bar and circle sit far apart, and the composition's circle
is behind RAP-MUSIC's.

## Runtime

Median seconds per trial over all 80 trials, on the same CPU host, OpenMEEG
limited to 8 OpenMP threads:

| method | median s |
| --- | --- |
| learned proposal alone | 0.0 |
| OpenMEEG dipole scan | 2.5 |
| RAP-MUSIC | 6.5 |
| proposal + physical refinement | 16.8 |
| uninformed initialization + refinement | 19.2 |
| four physical restarts | 69.0 |

This is **amortized inference time**: what it costs to run an estimator on one
epoch once the network exists. It does not include the cost of training the
network (a 200,000-step pre-training run, about 87 minutes on a GPU) or of
building the gain bank it trained on. Four physical restarts reach comparable
localization at about four times the inference cost of the composition.

## The demo trial

`h-k2-shared-close`, trial 4 — the trial the uninformed gradient arm failed worst
on, chosen by *that* arm's error and not by the composition's.

| | worst-source error | sensor residual |
| --- | --- | --- |
| uninformed initialization + refinement | 124.3 mm | 0.0117 |
| learned proposal alone | 8.7 mm | — |
| proposal + physical refinement | 6.9 mm | 0.0120 |

The two refinements are the same objective, the same optimizer, the same 300
steps and the same closed-form warm start for the dipole orientations. They
differ only in where they start.

**Near-equal sensor residual does not establish a unique anatomical solution.**
The anatomically poor answer fits the measurement very slightly *better*. At K=2
on rank-one data, two anatomically very different source configurations produce
nearly the same 64-channel topography, so the data fit does not identify the
sources and the result is sensitive to initialization. This is ambiguity and
non-identifiability plus initialization sensitivity. It is not a demonstration
that one run found a local minimum and the other found a global one; no global
optimum is computed anywhere in this repository.

## The composed derivative

One `jax.grad` of the composed objective, on the real OpenMEEG BEM, checked
against central differences of the same composed forward along 3 random
directions in the 1,146,796-dimensional parameter space.

| stage | derivative mechanism |
| --- | --- |
| network | `torch.autograd`, inside the `proposal` Tesseract |
| outer objective | JAX, in the orchestrator, which imports neither |
| source moment | hand-written analytic algebra, inside the `headfield` Tesseract |
| source position | central differences through OpenMEEG's C++ `DipSourceMat` assembly, inside the `headfield` Tesseract |

A composed directional finite-difference check verifies the packaged derivative
path; the smallest recorded relative discrepancy in the configured check is
2.51e-10 (`results/hybrid/gradcheck.json`, regenerate with `make gradcheck`).
This tests derivative composition and says nothing about localization accuracy.

Blocking the solver's position sensitivity changes the weight gradient by 3.2%,
so the compiled solver's derivative is inside the composed gradient rather than
attached alongside it.

Not every operation in the proposal is differentiated. The network emits a logit
and a continuous offset per lattice voxel; selecting which voxels become sources
is a hard `argmax` with greedy non-maximum suppression, which is discrete and
detached. The returned coordinates are the selected voxels' centres plus their
own continuous offsets, and those — with the dipole directions and everything
downstream — carry gradient.

## Scope of these numbers

- Synthetic observations only. No real EEG appears in this repository.
- Template anatomy (fsaverage). It is not anyone's head.
- `K` is given to every method.
- One training run, one configuration, one seed. No architecture search.
- The proposal's training distribution spans the generator's skull conductivity —
  the gain bank's head models cover a brain-to-skull ratio from 25:1 to 91:1 — so
  part of its advantage may be robustness to conductivity rather than a better
  starting point. The refinement runs on the same 50:1 ico3 operator the
  uninformed arm uses.
- The proposal network is trained for one specific 64-channel montage and one
  template head, both baked into its checkpoint.
- No clinical or medical-device claim is made or implied.
