# Rigorous Memory-Branch Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Remove statistical and finite-precision confounds from E12-S/E13-B/E13-C, and add a manifest-driven E14 trajectory-OT pullback runner that is executable on 16GB hardware.

**Architecture:** Preserve the released BF16 model as the deployment surface. Use balanced two-way decomposition for matched memory/noise branches, quantization-aware norm matching for controls, pair-level inference for oracle transplant results, and a NumPy OT target that is pulled back through the validated finite-response operator.

**Tech Stack:** Python 3.10+, NumPy, JAX (optional runtime), PyYAML, pytest.

**Spec:** `docs/2026-09-09-deployed-finite-response-and-readout-branching.md`

## Global Constraints

- The released BF16 policy remains unchanged.
- Current RGB, current robot state, instruction, model weights, and declared noise seeds remain fixed within paired interventions.
- FP32 JVP is diagnostic only; deployed edits use observed BF16 finite responses.
- 16GB runs process one state at a time and sequentially evaluate controls.
- Statistical resampling occurs at the pair/state level, never by treating two directions from one pair as independent.

---

### Task 1: Matched two-way memory/noise decomposition

**Files:**
- Modify: `src/trajmem_ot/memory_views.py`
- Modify: `scripts/run_e13_readout_mask_branching.py`
- Modify: `scripts/analyze_e13_readout.py`
- Test: `tests/test_memory_views.py`

**Interfaces:**
- Produces: `matched_variance_decomposition(actions) -> MatchedVarianceDecomposition`.

- [x] Write failing tests for additive memory/noise effects and interaction-only effects.
- [x] Verify the tests fail because the matched decomposition is absent.
- [x] Implement orthogonal balanced two-way decomposition.
- [x] Preserve legacy fields in JSON and add matched main-effect/interaction fields.
- [x] Aggregate matched fractions with state-level bootstrap intervals.
- [x] Run focused and full tests.

### Task 2: Quantization-aware E12-S controls and nonlinear fidelity

**Files:**
- Modify: `src/trajmem_ot/finite_response.py`
- Modify: `scripts/run_e12_secant_recovery.py`
- Modify: `scripts/analyze_e12_secant.py`
- Test: `tests/test_finite_response.py`

**Interfaces:**
- Produces: `quantized_norm_matched_memory(memory, direction, target_norm, ...)`.

- [x] Write a failing BF16 test requiring the applied control norm to match the target norm.
- [x] Verify the test fails because no quantization-aware matcher exists.
- [x] Implement bounded scale search and diagnostics.
- [x] Add configurable multiple random controls, fresh-noise controls, explicit corruption seeds, and actual-vs-predicted response metrics.
- [x] Aggregate pairwise ours-minus-random effects with state-level bootstrap and sign tests.
- [x] Run focused and full tests.

### Task 3: Context-quality and pair-level E13-B inference

**Files:**
- Create: `src/trajmem_ot/pair_quality.py`
- Create: `configs/e13_pair_quality.yaml`
- Modify: `scripts/audit_e13_pairs.py`
- Modify: `scripts/run_e13_oracle_transplant.py`
- Modify: `scripts/analyze_e13_oracle.py`
- Test: `tests/test_pair_quality.py`
- Test: `tests/test_hypothesis_metrics.py`

**Interfaces:**
- Produces: quality tiers, pair-level bootstrap intervals, headroom recovery, normalized memory-effect ratios, and tolerance-sensitivity curves.

- [x] Write failing tests for strict/moderate pair eligibility and headroom recovery.
- [x] Verify expected failures.
- [x] Implement pair-quality criteria and YAML loading.
- [x] Add quality annotations to the audit and optional runner enforcement.
- [x] Add coverage curves over predeclared tolerance multipliers.
- [x] Aggregate by pair and quality tier, with bootstrap/sign-test statistics.
- [x] Run focused and full tests.

### Task 4: Native-support coverage for E13-C

**Files:**
- Modify: `scripts/run_e13_readout_mask_branching.py`
- Modify: `scripts/analyze_e13_readout.py`
- Test: `tests/test_hypothesis_metrics.py`

**Interfaces:**
- Consumes independent native-history reference/target samples.
- Produces matched-compute native-support coverage gain for each readout allocation.

- [x] Add tests for calibrated tolerance and flattened candidate-set coverage.
- [x] Add `--calibrate-native-support` and independent seed blocks.
- [x] Report coverage gain relative to `(B,N)=(1,total_budget)`.
- [x] Run tests.

### Task 5: Manifest-driven E14 trajectory OT pullback

**Files:**
- Create: `src/trajmem_ot/transport.py`
- Create: `scripts/run_e14_ot_pullback.py`
- Create: `scripts/analyze_e14.py`
- Test: `tests/test_transport.py`
- Modify: `tests/test_scripts_smoke.py`

**Interfaces:**
- Consumes: JSON rollout manifest with `noise_seed` and scalar `return`.
- Produces: return-tilted OT transport, finite-response BF16 edit, same-noise controls, and saved action/memory artifacts for paired simulator execution.

- [x] Write failing tests for Sinkhorn marginals and transport toward the higher-return particle.
- [x] Verify failures.
- [x] Implement stable NumPy return tilting, cost, Sinkhorn, and barycentric transport.
- [x] Implement a 16GB sequential multi-noise finite-response runner.
- [x] Add aggregate analyzer and script smoke coverage.
- [x] Run focused and full tests.

### Task 6: Documentation and handoff

**Files:**
- Modify: `README.md`
- Modify: `RUN_NEXT.md`
- Modify: `results/STATUS.md`
- Create: `docs/2026-09-09-rigorous-next-experiments.md`

- [x] Correct the interpretation of the legacy E13-C `0.655` fraction.
- [x] Record E12-S as corruption recovery, not policy improvement.
- [x] Add exact 16GB commands for E13-B2, E13-C2, E12-S2, and E14-A.
- [x] Add stop/continue criteria and forbidden claims.
- [x] Run compileall, all tests, and all script `--help` checks.


## Verification record

- Focused new and compatibility tests: `35 passed`.
- `python -m compileall -q src scripts tests`: passed.
- All modified/new experiment scripts expose `--help` without the RoboMME runtime.
- Released-checkpoint GPU experiments remain the next execution step in `RUN_NEXT.md`.
