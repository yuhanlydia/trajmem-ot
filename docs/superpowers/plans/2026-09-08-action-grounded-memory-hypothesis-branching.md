# Action-Grounded Memory-Hypothesis Branching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a tested JAX action-response operator, structured memory bases, memory-view diagnostics, and 16GB/24GB experiment presets while preserving the existing TrajMem-OT baseline.

**Architecture:** The existing PyTorch OT core remains unchanged. New NumPy/JAX modules isolate memory-basis construction, exact action JVPs, SVD-regularized pullback, and memory-versus-noise uncertainty analysis. A runtime-only RoboMME adapter lets the real checkpoint run in-process without making upstream packages mandatory for ordinary unit tests.

**Tech Stack:** Python 3.10+, NumPy, PyTorch, optional JAX, pytest, RoboMME/openpi at runtime.

**Spec:** `docs/superpowers/specs/2026-09-08-action-grounded-memory-hypothesis-branching-design.md`

## Global Constraints

- Only historical memory may be edited; current image, state, instruction, parameters, and chosen noise are immutable.
- 16GB preset: 8 basis directions, JVP chunk 2, 4 views, 2 noises/view, rank 4, XLA fraction 0.75.
- 24GB preset: 16 basis directions, JVP chunk 4, 8 views, 4 noises/view, rank 8, XLA fraction 0.88.
- The package must import without RoboMME installed.
- Tests must run on CPU.
- Old E7/E8 scripts and results are retained.

---

### Task 1: Compute presets

**Files:**
- Create: `src/trajmem_ot/presets.py`
- Test: `tests/test_presets.py`

**Interfaces:**
- Produces: `ComputePreset`, `get_compute_preset(name: str) -> ComputePreset`.

- [ ] Write tests asserting exact 16GB/24GB values and rejection of unknown presets.
- [ ] Run `PYTHONPATH=src pytest tests/test_presets.py -q` and observe failure because the module is absent.
- [ ] Implement immutable validated presets.
- [ ] Re-run the test and the full suite.

### Task 2: Structured memory bases

**Files:**
- Create: `src/trajmem_ot/memory_basis.py`
- Test: `tests/test_memory_basis.py`

**Interfaces:**
- Produces: `random_rank_one_basis`, `history_difference_basis`, `combine_basis`, `normalize_basis`.

- [ ] Write tests for unit norms, deterministic seeds, orthogonality of history-SVD basis, and coefficient reconstruction.
- [ ] Run the focused test and observe the missing-module failure.
- [ ] Implement NumPy basis functions with shape validation.
- [ ] Re-run focused and full tests.

### Task 3: Exact JAX action-response operator

**Files:**
- Create: `src/trajmem_ot/jax_operator.py`
- Test: `tests/test_jax_operator.py`

**Interfaces:**
- Produces: `action_jvp`, `batched_action_jvps`, `central_action_secant`, `response_svd`, `svd_ridge_pullback`, `energy_rank`.

- [ ] Write nonlinear toy-policy tests showing JVP/secant agreement and chunked/single JVP equality.
- [ ] Write a linear inverse test showing SVD-ridge pullback reduces target residual and respects rank truncation.
- [ ] Run focused tests and observe failure because the API is absent.
- [ ] Implement optional-JAX functions and NumPy SVD utilities.
- [ ] Re-run focused and full tests.

### Task 4: Memory-view branching diagnostics

**Files:**
- Create: `src/trajmem_ot/memory_views.py`
- Test: `tests/test_memory_views.py`

**Interfaces:**
- Produces: `build_memory_views`, `variance_decomposition`, `nearest_target_coverage`, `allocation_grid`.

- [ ] Write tests for trust-radius bounded views and exact total-variance decomposition.
- [ ] Write a synthetic test where between-memory variance exceeds within-noise variance.
- [ ] Run focused tests and observe failure.
- [ ] Implement the branching and metrics.
- [ ] Re-run focused and full tests.

### Task 5: Optional RoboMME in-process adapter

**Files:**
- Create: `src/trajmem_ot/robomme_jax.py`
- Create: `tests/test_robomme_jax.py`

**Interfaces:**
- Produces: `replace_observation_memory`, `build_fixed_noise_action_problem`, `prepare_policy_observation`.

- [ ] Write dummy-struct tests proving only the selected memory field changes and fixed noise is reused.
- [ ] Run focused tests and observe failure.
- [ ] Implement duck-typed helpers with upstream imports inside functions only.
- [ ] Re-run focused and full tests.

### Task 6: Runnable experiment entry points

**Files:**
- Create: `scripts/run_e11_jvp_smoke.py`
- Create: `scripts/run_e11_robomme_jvp.py`
- Create: `scripts/run_e12_operator_basis.py`
- Create: `scripts/run_e13_memory_view_branching.py`
- Create: `scripts/analyze_e11.py`
- Create: `scripts/bootstrap_robomme.sh`
- Create: `configs/16gb.yaml`
- Create: `configs/24gb.yaml`

**Interfaces:**
- E11 smoke runs without RoboMME.
- E11 real runner accepts checkpoint, data, index, preset, output, seed, and finite-difference radii.
- E12 runs clean/degraded memory recovery in the same local checkpoint process.
- E13 runs matched-compute memory-view/noise allocations.

- [ ] Add CLI parser tests for hardware presets and required paths.
- [ ] Implement CPU smoke and JSON output schemas.
- [ ] Implement the in-process RoboMME runner using the runtime adapter.
- [ ] Implement an idempotent upstream bootstrap script pinned to a documented commit.
- [ ] Run smoke tests and full pytest.

### Task 7: Documentation and handoff

**Files:**
- Modify: `README.md`
- Modify: `results/STATUS.md`
- Create: `RUN_NEXT.md`
- Modify: `src/trajmem_ot/__init__.py`
- Modify: `pyproject.toml`

- [ ] Document the current old results without relabeling them.
- [ ] Document E11–E14 and exact 16GB/24GB commands.
- [ ] Export stable public APIs and add optional JAX/dev dependencies.
- [ ] Run `PYTHONPATH=src pytest -q`.
- [ ] Run `python scripts/run_e11_jvp_smoke.py --preset 16gb` and `--preset 24gb`.
- [ ] Verify no generated data, checkpoints, or third-party repositories are committed.
