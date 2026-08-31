.PHONY: help setup demo test test-images build gen-tests \
        summary figures hero hero-data cortex-mesh \
        gradcheck batching vjp-report observations benchmark clean

VENV ?= .venv
PY   ?= $(VENV)/bin/python
# The Tesseract CLI ships with tesseract-core inside the venv; don't rely on it
# being on PATH. Override TESS=tesseract if you have it installed globally.
TESS ?= $(VENV)/bin/tesseract
TESSERACTS := headfield proposal

# OpenMEEG's OpenMP scaling is pathological on a many-core host: the per-step
# source assembly is 67x slower at 48 threads than at 8. Do not remove this.
THREAD_ENV = OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8

#: The benchmark sweep runs as several 8-thread processes rather than one wide
#: one, for the same reason.
JOBS    ?= 6
THREADS ?= 8

CONDITIONS = $(shell $(PY) -c "import sys; sys.path.insert(0,'app'); \
	from neurolayout.hybrid.benchmark import CONDITIONS as C; print(' '.join(c.name for c in C))")

help:
	@echo "NeuroLocate — differentiable EEG inference across PyTorch, OpenMEEG and JAX"
	@echo ""
	@echo "  make setup        Create $(VENV) and install the app, components and dev deps"
	@echo "  make demo         One hard K=2 inversion through all three layers (~1 min)"
	@echo "  make test         The full in-process test suite (no container runtime needed)"
	@echo "  make build        Build both Tesseract images (needs Docker)"
	@echo "  make test-images  Run the packaged test cases against the built images"
	@echo ""
	@echo "Verifying the frozen numbers:"
	@echo "  make summary      Recompute the K=2 tables from the committed shards"
	@echo "  make gradcheck    Re-run the composed derivative check (~2 min)"
	@echo "  make batching     What the batched component path costs"
	@echo "  make vjp-report   Source-derivative audit and step-size sweep"
	@echo ""
	@echo "Regenerating artifacts (all optional; the committed ones are enough):"
	@echo "  make observations Rebuild the benchmark EEG from the MNE ico4 generator"
	@echo "  make benchmark    Re-run every method over the matrix ($(JOBS)x$(THREADS) threads)"
	@echo "  make figures      The architecture and benchmark figures"
	@echo "  make hero         The README animation, from the recorded trajectories"
	@echo "  make hero-data    Replay the frozen K=2 trial to record those trajectories"
	@echo "  make cortex-mesh  The cortical surface the animation is drawn on (needs MNE)"
	@echo "  make gen-tests    Regenerate the packaged component test cases"

setup:
	@command -v uv >/dev/null 2>&1 || { echo "uv not found: https://docs.astral.sh/uv/"; exit 1; }
	uv venv --python 3.11 $(VENV)
	uv pip install --python $(PY) -e components/shared_code
	uv pip install --python $(PY) -e 'app[dev]'
	uv pip install --python $(PY) --index-url https://download.pytorch.org/whl/cpu torch
	@echo "Done. Verify with: make demo"

#: The judge-facing demo. Needs nothing but `make setup`: the trained proposal
#: network is packaged inside its component and the observation artifact is
#: committed, so this runs offline against real OpenMEEG.
demo:
	env OMP_NUM_THREADS=8 JAX_PLATFORMS=cpu $(PY) scripts/demo.py --json results/demo.json

test:
	$(PY) -m pytest

#: Recompute every number docs/BENCHMARK.md quotes, from the committed shards.
summary:
	$(PY) scripts/summarize_benchmark.py

gradcheck:
	env $(THREAD_ENV) $(PY) scripts/report_hybrid_gradcheck.py

batching:
	env $(THREAD_ENV) $(PY) scripts/report_batched_headfield.py

vjp-report:
	env $(THREAD_ENV) $(PY) scripts/report_source_vjp.py

#: Needs MNE. Writes results/hybrid/observations.npz, which is committed.
observations:
	$(PY) scripts/build_hybrid_observations.py

#: Resumable at condition granularity: the runner skips a shard that already
#: exists, so re-running picks up where an interrupted sweep stopped.
benchmark:
	@mkdir -p results/hybrid/shards
	@echo "$(words $(CONDITIONS)) conditions, $(JOBS) processes x $(THREADS) threads"
	@printf '%s\n' $(CONDITIONS) | xargs -P $(JOBS) -I{} sh -c \
		'OMP_NUM_THREADS=$(THREADS) OPENBLAS_NUM_THREADS=$(THREADS) \
		 MKL_NUM_THREADS=$(THREADS) \
		 $(PY) scripts/run_hybrid_benchmark.py --conditions {} \
			--methods rapmusic scan gradient gradient_restarts proposal hybrid \
			> results/hybrid/shards/{}.log 2>&1 \
			&& echo "  done {}" || echo "  FAILED {} (see results/hybrid/shards/{}.log)"'
	@echo "shards finished; re-run to resume any that died, then 'make summary'"

#: The two static figures. Both read the committed shards and nothing else.
figures:
	$(PY) scripts/plot_architecture.py
	$(PY) scripts/plot_benchmark.py

#: Needs MNE and the fsaverage anatomy. The output is committed, so this is only
#: needed if the mesh itself has to be rebuilt.
cortex-mesh:
	$(PY) scripts/build_cortex_mesh.py

#: Replays one frozen deterministic trial of `h-k2-shared-close` to record the
#: two optimizer trajectories the shards do not store, and refuses to write
#: anything unless every reproduced error and residual matches the frozen shard.
#: About five minutes. `hero` needs only what it wrote, plus ffmpeg (or the
#: `imageio-ffmpeg` wheel) for the MP4.
hero-data:
	env $(THREAD_ENV) $(PY) scripts/build_hybrid_k2_visual.py

hero:
	$(PY) scripts/plot_hero.py

gen-tests:
	$(PY) scripts/gen_test_cases.py

# --- containerized path (requires a Docker-compatible runtime) ---------------

build:
	@for name in $(TESSERACTS); do \
		echo "Building $$name..."; \
		$(TESS) build components/tesseracts/$$name || exit 1; \
	done

test-images:
	@for name in $(TESSERACTS); do \
		dir="components/tesseracts/$$name/test_cases"; \
		if [ ! -d "$$dir" ]; then \
			echo "ERROR: $$name has no packaged test_cases directory"; \
			exit 1; \
		fi; \
		found=0; \
		for case_file in "$$dir"/*.json; do \
			[ -f "$$case_file" ] || continue; \
			found=1; \
			echo "  $$case_file"; \
			$(TESS) run neurolayout_$$name test @$$case_file || exit 1; \
		done; \
		if [ "$$found" -eq 0 ]; then \
			echo "ERROR: $$name has no packaged JSON test cases"; \
			exit 1; \
		fi; \
	done

clean:
	rm -rf $(VENV) .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
