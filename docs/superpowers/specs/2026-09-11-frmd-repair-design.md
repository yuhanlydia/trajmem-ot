# FRMD repair core: implementation from the supplied protocol

The remote revision e901e44 did not contain the previously described E14-R
implementation. This is a new implementation of the user's supplied method,
not a recovered copy of that absent commit.

The first deliverable is a reusable memory-only distillation engine and a real
GPU corruption smoke. Native interference manifests, persistence and physical
evaluation are subsequent integrations, not claims made by the smoke.

## Method

Use one frozen policy with student and temporary-teacher history conditioning.
The primary target for each optimization noise is teacher action minus student
action under that same noise. Also expose weighted-centroid and cross-view OT
barycentric targets, without assuming OT is better.

At each repair iteration, quantize student memory plus/minus each normalized
basis direction using FP32 endpoint arithmetic followed by deployment BF16
quantization. Divide action half-chords by the norm of the actual FP32
half-difference between the two quantized memory endpoints. Stack all
optimization-noise action responses and solve the small response matrix by
truncated ridge SVD. Never decompose the complete memory or model parameters.

Defaults: K=8, response rank=4, relative probe radius=2.5e-4, two repair
iterations, per-step relative radius=1.25e-3, total radius=2.5e-3, optimization /
validation / evaluation noise counts=8/4/8, alpha candidates=(0.25,0.5,1.0).
Radii use the initial quantized student norm as the trust-region reference;
probe radii use the current memory norm. Enforce radii on applied quantized
updates. A quantization-erased column is retained as a zero response with an
explicit diagnostic, rather than silently changing K or crashing a case.

Rebuild finite responses at the current memory on every iteration. Validation
noises select the step and accept only positive median squared-error reduction
with an explicitly configured improving-noise fraction (default 0.5). Rejecting
an edit rolls back that iteration; it does not stop the experiment program.
Final evaluation noises never participate in targets, direction selection,
step selection or acceptance. Count every actual policy trajectory call.

## Experimental policy update from the user

The earlier +5 percentage-point success and 25% repair-ratio thresholds are
no longer automatic stop/continue gates. Record metrics and controls, inspect
positive trends and reusability, debug and refine unsuccessful settings. Do
not call action alignment task success, synthetic corruption real interference,
or exploratory improvements statistically established results.

## Interfaces

- `self_teacher.distillation_targets(student_actions, teacher_actions, mode,
  teacher_weights=None, epsilon=None)`: aligned residual array.
- `repair_acceptance.assess_repair(before_losses, after_losses,
  minimum_improving_fraction=0.5)`: serializable acceptance evidence.
- `finite_distillation.NoisePartition`: three nonempty disjoint integer-seed
  tuples, rejecting overlap or duplicates before any policy calls.
- `finite_distillation.RepairConfig`: validated algorithm parameters.
- `finite_distillation.repair_memory(student_memory, basis, student_action,
  teacher_action, quantize, noises, config)`: repaired memory, per-step trace,
  fresh-noise evaluation and actual query accounting. `student_action` accepts
  `(memory, noise_seed)`; `teacher_action` accepts `noise_seed`. Backend-specific
  policy/state preparation stays outside this core.

## Verification

Use deterministic nonlinear toy policies to demonstrate relinearization and
improvement across iterations; a conflicting validation view to demonstrate
rollback; coarse quantization to verify actual chord normalization and applied
radius bounds; a call recorder to verify final-noise isolation and exact probe
counts. Test paired/centroid/OT targets independently. Then run real frozen
RoboMME BF16 smoke with explicit synthetic corruption labels and controls.
