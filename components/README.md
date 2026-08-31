# Components

Two Tesseracts and the NumPy-only code they share.

| Component | Stack | Derivative | Role |
| --- | --- | --- | --- |
| [`tesseracts/proposal`](tesseracts/proposal) | PyTorch (CPU wheels) | `torch.autograd` VJP | sensor covariance + network weights → `K` source positions |
| [`tesseracts/headfield`](tesseracts/headfield) | NumPy + OpenMEEG (C++) | hand-written analytic and finite-difference VJP | source positions + moments → 64-channel EEG |
| [`shared_code`](shared_code) | NumPy only | — | geometry, the cached OpenMEEG head model, the sphere solver, the source-model algebra |

`shared_code` must not import JAX or PyTorch. The whole reason the components are
split is that they do not share an AD stack, and a shared dependency on one would
quietly undo that.

`headfield` serves three modes behind one schema: `localize` (one source set),
`localize_batch` (many source sets in one call) and `montage` (the analytic
sphere forward, kept as an independent check on the BEM and as a fallback that
needs no cached anatomy). Its `backend` field selects OpenMEEG or the sphere.

## Working on a component

```bash
make test                      # runs everything, in-process, no Docker
make build                     # build both images (needs Docker)
make test-images               # frozen test cases against built images
make gen-tests                 # regenerate test_cases/*.json after intended changes
```

Each component's `test_cases/*.json` freezes an endpoint's inputs and outputs.
They are executed both in-process (`tests/test_component_cases.py`) and against
built images (`make test-images`), so a behaviour change fails loudly in either
transport.
