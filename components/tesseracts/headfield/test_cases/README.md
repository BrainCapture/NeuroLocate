# Frozen test cases — `headfield`

Each `*.json` file pins one endpoint's inputs to its exact outputs (or its
expected exception). They are run two ways:

- in-process, by `tests/test_component_cases.py` (`make test`);
- against the built image, by `make test-images`
  (`tesseract run neurolayout_headfield test @<file>`).

Regenerate with `make gen-tests` — but only when a behaviour change is intended,
and review the diff. These are the determinism anchors for the analytic solver
and the hand-written VJP; a silent numeric drift here is exactly what they exist
to catch.
