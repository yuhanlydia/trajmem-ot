# FRMD Repair Implementation Plan

> For agentic workers: execute inline with the existing test-driven workflow.
> Steps use checkbox syntax; continue debugging within the user's authorized scope.

**Goal:** Implement and execute the supplied finite-response repair mechanism.
**Architecture:** Pure NumPy orchestration around injected deployment quantization
and frozen-policy callbacks; a separate RoboMME CLI prepares observations and noise.
**Tech Stack:** Python, NumPy, existing ridge/Sinkhorn utilities, JAX/BF16 runtime.
**Spec:** ../specs/2026-09-11-frmd-repair-design.md

## Global Constraints

- No weight updates, LoRA, SFT, external teacher or BF16 JVP edits.
- Optimization, validation and evaluation noise seeds must be disjoint.
- Defaults K=8, r=4, h=2.5e-4, iterations=2, step radius=1.25e-3,
  total radius=2.5e-3, noise counts=8/4/8, alpha=(0.25,0.5,1.0).
- Preserve evidence and controls; numerical trend thresholds do not stop the program.
- Synthetic GPU smoke is not a native-interference or physical benchmark result.

## Task 1: Targets and acceptance

Files: `src/trajmem_ot/self_teacher.py`, `repair_acceptance.py`;
`tests/test_self_teacher.py`, `tests/test_repair_acceptance.py`.

- [x] Write failing tests with `student=[0,10]`, `teacher=[2,12]`:
  paired targets are `[2,2]`; centroid targets with weights `[.25,.75]`
  are `[9.5,-.5]`. Reject mismatched shapes and invalid weights.
- [x] Verify OT on identical, well-separated particles is near the identity
  residual and that transport retains source rows and teacher weights.
- [x] Check acceptance on before `[2,2,2,2]`, after `[1,1,3,3]` is false
  (median improvement zero), and after `[1,1,1,3]` is true.
- [x] Implement aligned subtraction, weighted teacher means and existing
  log-domain Sinkhorn barycentric residuals; validate all numeric inputs.
- [x] Run the two test files and retain the results.

## Task 2: Iterative quantized repair

Files: `src/trajmem_ot/finite_distillation.py`,
`tests/test_finite_distillation.py`.

- [x] Test seed overlap rejection before the first callback.
- [x] With `F(m)=m*m`, student `[1]`, teacher `[2]`, identity quantization,
  basis `[[1]]`, alpha `[1]` and large trust radii, assert the second
  linearization is at the accepted first memory and two steps beat one.
- [x] Test conflicting validation targets reject edits and preserve memory.
- [x] Test quantized endpoints and applied step/total radius enforcement.
- [x] Assert endpoint policy calls equal `2*K*Nopt*iterations`; evaluation
  seeds are first seen only after all repair/selection calls finish.
- [x] Implement validated dataclasses, injected callbacks, cached teacher
  targets, endpoint stacking, ridge solve, quantized trust-region backtracking,
  alpha selection, rollback traces, fresh evaluation and counters.
- [x] Run focused tests, then the repository suite.

## Task 3: Real GPU smoke runner

Files: `scripts/run_frmd_smoke.py`, `configs/frmd_smoke_16gb.yaml`,
`tests/test_frmd_smoke_cli.py`.

- [x] Write a subprocess dry-run test that loads no checkpoint and checks
  256 two-sided trajectory probes for the default case configuration.
- [x] Add arguments for checkpoint/data/index, synthetic row-corruption mode,
  target, basis source, output, noise counts and iteration count.
- [x] Prepare frozen policy callbacks using the existing RoboMME runtime and
  fixed-noise adapter; quantize endpoint memory to BF16 explicitly.
- [x] Use independent rank-one directions by default; accept an explicitly
  supplied history basis with recorded provenance. Do not inject the exact
  corruption-reversal direction as an undisclosed basis shortcut.
- [x] Evaluate baseline, teacher, repaired, negative, applied-norm-matched
  random and direct-interpolation controls under fresh noise. Report control
  matching error instead of silently treating unmatched norms as matched.
- [x] Run CLI help, dry-run, full tests and real GPU smoke; save JSON and NPZ
  locally and publish only lightweight, accurately labeled results.

## Task 4: Next integrations

After the working repair core and smoke, prepare a separate integration plan
for provenance-validated native history concatenation, per-task pilot cases,
persistence under later query observations, and paired physical execution.
Keep these incomplete until their actual code and evidence exist.
