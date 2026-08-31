# Frozen test cases — `proposal`

Each `*.json` file pins one endpoint's inputs to its exact outputs. They are run
two ways:

- in-process, by `tests/test_component_cases.py` (`make test`);
- against the built image, by `make test-images`
  (`tesseract run neurolayout_proposal test @<file>`).

Both cases run the **packaged** `proposal.pt` with its trained weights — no
`checkpoint` name and no `weights` array — so they are the determinism anchor for
the checkpoint that ships inside the image.

What each file fixes:

- `apply.json` — one epoch, `[1, 64, 1]`, with `n_sources = 2`. All four outputs
  are pinned: `positions_m`, `moments`, `scores` and `count_logits`. The count
  head is checked even though `n_sources` is supplied, because it is the output
  nothing else in the repository reads and would otherwise drift unnoticed.

  `T = 1` is the shortest valid epoch: the network's only view of the data is the
  sensor covariance `Y Yᵀ`, which one sample already defines. The epoch is
  unit-scale rather than volt-scale for the same reason — the covariance is
  normalized by its own trace, so amplitude is not part of the input, and
  unit-scale numbers keep the tolerances meaningful instead of vacuous.

- `vector_jacobian_product.json` — `dL/d(eeg)` for a fixed cotangent on
  `positions_m`.

  The VJP is taken with respect to `eeg` and **not** `weights`. The derivative
  with respect to the flattened parameters is the scientifically load-bearing one
  — it is what makes the network trainable through the BEM — but it is 1.15 M
  numbers, and freezing it here would mean a hundred-megabyte JSON for no extra
  coverage of the VJP machinery. Gate M is where `dL/dweights` is checked, across
  both component boundaries and against central differences.

Regenerate with `make gen-tests` — but only when a behaviour change is intended,
and review the diff. A silent numeric drift here is exactly what these exist to
catch.
