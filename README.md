# TrajMem-OT

**Action-grounded memory-hypothesis branching and trajectory-to-memory transport for memory-augmented VLAs.**

This repository contains the original frozen-policy trajectory-memory experiments and the next research stage built around an exact JAX memory-to-action operator.

## Research question

A diffusion VLA normally encodes one interaction history into one memory state and then samples many actions by changing only diffusion noise:

\[
M_t=E(H_t),\qquad A_n=F_\theta(M_t,\epsilon_n).
\]

Those samples explore action uncertainty **conditioned on one shared interpretation of history**. In memory-dependent tasks, the missing uncertainty can instead be over competing interpretations of the past. More diffusion samples can therefore agree with one another while all inheriting the same memory error.

TrajMem-OT separates the two axes:

\[
A_{b,n}=F_\theta(M_t^{(b)},\epsilon_n),
\]

where `b` indexes a coherent memory hypothesis/view and `n` indexes conditional action noise. It then uses exact JAX Jacobian-vector products to retain only memory directions that can actually control the action policy:

\[
Y=J_M D,\qquad J_M=\frac{\partial A}{\partial M}.
\]

Given a desired action-space change `u`—from a clean-memory self-teacher, a return-weighted target, or trajectory OT—the structured memory update is solved as

\[
c^*=\arg\min_c\|Yc-u\|_2^2+\lambda\|c\|_2^2,
\qquad \Delta M=Dc^*.
\]

SVD is applied to the **action-response matrix** `Y`, not repeatedly to the full raw memory tensor.

## What is implemented

### Reproducible original baseline

- return-tilted empirical trajectory distribution;
- log-domain Sinkhorn OT and barycentric action-trajectory transport;
- generic matrix-free JVP/VJP normal operator with conjugate gradient;
- low-rank, temporal-sparse, trust-region memory updates;
- strict `M`, `M+`, `M-`, norm-matched-random controls;
- deterministic RoboMME branch reconstruction and paired analysis;
- the original 64-state finite-difference E7 results and compact E8 results.

### Latest direction

- `16gb` and `24gb` compute presets;
- random rank-one and matched-history-difference memory bases;
- exact and chunked JAX JVPs for `memory -> action`;
- central-action-secant validation of exact JVPs;
- action-response SVD, effective-rank diagnostics, and ridge pullback;
- immutable RoboMME observation adapter that can edit history memory only;
- in-process released-checkpoint loader, avoiding the NumPy/WebSocket autodiff break;
- reward-free clean/degraded-memory recovery (`E12`, currently gated);
- forward-only matched-compute memory-view versus diffusion-noise branching
  (`E13-A`);
- target-mode coverage and total-variance decomposition.

The released-checkpoint E11–E13 experiments require an NVIDIA GPU and the upstream RoboMME/OpenPI environment. The released checkpoint and the 80-episode sample have now been downloaded locally; eight real E11 verification states were evaluated on a 16GB RTX A4000. CPU mathematical tests and synthetic JAX smoke experiments remain available for fast regression checks.

## Repository layout

```text
configs/                         16GB and 24GB experiment presets
src/trajmem_ot/                 reusable OT, JAX, basis, view, and RoboMME adapters
scripts/run_e11_jvp_smoke.py    hardware-independent JAX operator smoke test
scripts/run_e11_robomme_jvp.py  exact released-checkpoint JVP/FD comparison
scripts/run_e11_stage_localization.py
                                memory encoder/modulation stage diagnostics
scripts/run_e12_operator_basis.py
                                reward-free memory-recovery experiment
scripts/run_e13_memory_view_branching.py
                                memory-view vs action-noise decomposition
scripts/build_history_difference_basis.py
                                coherent basis from matched histories
RUN_NEXT.md                     exact commands for the next GPU experiments
results/STATUS.md               evidence already established and open claims
```

## CPU verification

```bash
python -m pip install -e '.[dev,jax]'
pytest -q

python scripts/run_e11_jvp_smoke.py \
  --preset 16gb \
  --output results/e11_smoke_16gb.json
```

The smoke experiment checks that exact JVPs match central action secants and that the SVD-ridge inverse reduces its action-space residual.

## Released-checkpoint verification status

On 2026-09-08 the following setup was validated:

- RoboMME upstream pinned to `ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b`;
- CUDA JAX detected `cuda:0` and the 16GB synthetic smoke passed;
- the downloaded `perceptual-framesamp-modul/79999` checkpoint and
  `robomme_preprocessed_data_sample` directory passed the required structure checks;
- the repository test suite passed (`44 passed` after the BF16 tangent,
  quantization-aware chord, and channel-diagnostic additions).

The first released-checkpoint run exposed two environment/numerical issues. The
upstream import needed the system `libGL.so.1` runtime, and the real checkpoint
stores memory in BF16 while the generated basis starts in FP32. The former was
installed as a host dependency; the latter is handled in
`jax_operator.py` by casting tangents to the primal memory dtype (commit
`2e6cc03`).

The exact JVP path now runs end to end on all eight verification states. The
aggregate report has median warmed latency `1.074 s` and median
`effective_rank_90 = 2`; no state produced an OOM or runtime failure. This is
an execution result and a preliminary sampled-rank signal, not yet a claim
that the deployed action-controllable memory subspace has rank two.

Finite-difference comparisons at the original tiny radii are not a valid
success gate for this BF16 memory: the perturbations are below the
representable spacing and produce large secant error. Across eight states the
legacy protocol had median FD cosine `0.0142` and worst relative error `2713`.

E11-B was then run on the same eight states with the quantization-aware
actual-chord protocol. For each endpoint it explicitly forms BF16 `M+` and
`M-`, measures the actual half-step `(M+ - M-) / 2`, and computes the action
chord after converting action outputs to FP32. All 64 comparisons qualified
the endpoint diagnostics: the changed fraction was roughly 2.6%–19.3% and
the positive/negative asymmetry was below 0.062. The numerical comparison
still failed: the median qualified cosine was `-0.00046`, with maximum
relative error `1.0044`. The action chord norm was around `5e-3`, while the
local JVP response was typically `1e-6`–`1e-4`.

A same-memory repeat call with the same fixed noise had action difference norm
`0.0` on the rerun state, so the nearly radius-independent chord is not
explained by sampling nondeterminism in this harness.

This means endpoint quantization alone does not explain the discrepancy. It
exposes a stronger BF16/model-path nonlinearity or a remaining derivative
path mismatch. The qaware chord is therefore a diagnostic failure, not
evidence that the exact JVP is correct or incorrect by itself. The final
status is split explicitly:

- **Execution:** success — exact JVP, chunking, SVD, and rank diagnostics run on
  the released checkpoint and GPU.
- **Numerical validation:** failure/inconclusive under both the legacy and
  quantization-aware BF16 chord protocols. The endpoint diagnostics are
  observable and symmetric, but the chord does not agree with the local JVP.
  It must not be used to claim operator correctness.

E12 remains paused because it consumes the unvalidated JVP as a pullback
operator. E13-A is intentionally independent of JVP and now runs as a
forward-only phenomenon test using coherent history-difference views.

### E11-C localization and E13-A phenomenon check

E11-C was run on states 0 and 1 at `num_steps` 1, 2, 5, and 10, with two
directions per state. Every comparison reports the deployed robot channels
`A[..., :8]`, the padded channels `A[..., 8:32]`, and all 32 channels. The
chord midpoint `M_c=(M+ + M-) / 2` is also used for a second JVP comparison.
The median robot-channel cosine over the eight default runs was `-0.0058`
(`-0.0060` at the midpoint), essentially unchanged from the all-channel
diagnostic. At `num_steps` 1/2/5/10 the robot-channel medians were
`-0.019/-0.014/+0.039/-0.001`; there is no monotonic one-step rescue.

The JVP transformed primal is not equal to the plain forward: the recorded
`base_primal_delta_norm` ranges from `0.011` to `0.092` across these runs.
This is a directly measured numerical-path discrepancy and must be resolved
before interpreting the AD tangent as a deployed derivative. Setting
`JAX_DEFAULT_MATMUL_PRECISION=highest` for states 0/1 at 10 steps changed the
robot cosine median only to `0.0124` and did not remove the primal mismatch.

E13-A uses eight same-task/different-episode history pairs to build a coherent
history-difference basis and never calls JVP, SVD pullback, or E12. In the
real 8D robot channels, memory-variance fractions were `0.00019/0.00170` for
the `(B,N)=(2,4)/(4,2)` allocations on state 0 and `0.00023/0.00171` on
state 1. With this small edit radius and two states, diffusion noise still
dominates the observed action variance; this is a preliminary null result,
not an environment-success claim.

The stage-localization probe confirms this boundary on two states with one
flow step: `static_image_emb -> mem_tokens` has transformed-primal delta
`0.0`, while the first memory-modulation velocity has primal deltas `0.0612`
and `0.0565`, with velocity JVP/chord cosines about `0.063` and `0.061`.
This narrows the numerical failure to the modulation/LLM path. The remaining
stage decomposition into every internal transformer block has not been used
to make a scientific claim.

The saved E11-C and E13-A JSON reports include channel splits, midpoint
metrics, primal deltas, precision mode, and the exact allocation grid. No
task-success or reward improvement has been established.

## Bootstrap the real RoboMME runtime

The helper pins the public upstream repository to a tested source revision and installs this package into the same `uv` environment:

```bash
bash scripts/bootstrap_robomme.sh
```

Optional simulator submodule setup:

```bash
ROBOMME_WITH_SIM=1 bash scripts/bootstrap_robomme.sh
```

Prepare the released `perceptual-framesamp-modul/79999` checkpoint and official preprocessed sample data according to the upstream RoboMME instructions. Then follow [`RUN_NEXT.md`](RUN_NEXT.md).

## Compute policy

The default 16GB configuration uses one environment state at a time, eight memory-basis directions in JVP chunks of two, four memory views, and two noises per view. The 24GB configuration uses 16 directions in chunks of four, eight views, and four noises per view. Batching is deliberately spent on JVP directions and diffusion particles that share model weights and the current observation, rather than on unrelated episodes.

## Current scientific status

The existing results establish that history memory is a causal control variable of the frozen MME-VLA: a history-only intervention changes its action chunk, and the changed action produces a distinct physical simulator future. The old 16-direction scalar finite-difference estimator was weak and heterogeneous; it did **not** test the exact JAX operator or memory-hypothesis branching. Environment-return improvement has not yet been established. See [`results/STATUS.md`](results/STATUS.md).
