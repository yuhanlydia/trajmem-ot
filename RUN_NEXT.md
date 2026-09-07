# Next GPU Experiments

This is the execution handoff for an agent working on an NVIDIA 16GB or 24GB GPU. Run the stages in order. Do not rerun the old scalar finite-difference E7 as the main experiment.

## 0. Pull and verify

```bash
git pull --ff-only
python -m pip install -e '.[dev,jax]'
pytest -q
python scripts/run_e11_jvp_smoke.py --preset 16gb --output results/e11/smoke_16gb.json
```

For the released MME-VLA runtime:

```bash
bash scripts/bootstrap_robomme.sh
export TRAJMEM_ROOT="$(pwd)"
export ROBOMME_DIR="$TRAJMEM_ROOT/third_party/robomme_policy_learning"
export CHECKPOINT=/absolute/path/to/perceptual-framesamp-modul/79999
export DATA=/absolute/path/to/robomme_preprocessed_data_sample
mkdir -p "$TRAJMEM_ROOT/results/e11" "$TRAJMEM_ROOT/results/e12" "$TRAJMEM_ROOT/results/e13"
cd "$ROBOMME_DIR"
```

All following commands execute inside the upstream `uv` environment but call scripts from this repository.

## 1. E11 — exact JAX memory-to-action operator

### 16GB

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
for INDEX in 0 1 2 3 4 5 6 7; do
  uv run python "$TRAJMEM_ROOT/scripts/run_e11_robomme_jvp.py" \
    --checkpoint "$CHECKPOINT" \
    --data "$DATA" \
    --index "$INDEX" \
    --preset 16gb \
    --output "$TRAJMEM_ROOT/results/e11/state_${INDEX}_16gb.json"
done
uv run python "$TRAJMEM_ROOT/scripts/analyze_e11.py" \
  "$TRAJMEM_ROOT"/results/e11/state_*_16gb.json \
  --output "$TRAJMEM_ROOT/results/e11/summary_16gb.json"
```

### 24GB

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.88
for INDEX in 0 1 2 3 4 5 6 7; do
  uv run python "$TRAJMEM_ROOT/scripts/run_e11_robomme_jvp.py" \
    --checkpoint "$CHECKPOINT" \
    --data "$DATA" \
    --index "$INDEX" \
    --preset 24gb \
    --output "$TRAJMEM_ROOT/results/e11/state_${INDEX}_24gb.json"
done
```

Inspect:

- JVP–central-secant cosine across radii;
- relative secant error;
- first-call and warmed latency;
- `effective_rank_90` and singular-value decay;
- peak GPU memory externally with `nvidia-smi`.

A poor JVP–secant match indicates an implementation/numerical problem. A low response rank is a scientific observation about the local action-controllable memory subspace.

## 2. Build a coherent history-difference basis

Create a JSON file whose entries are matched dataset indices, for example:

```json
[
  {"a": 120, "b": 124},
  {"a": 813, "b": 817}
]
```

Pairs should have the same task/current decision context but differ in the task-critical history variable. Then run:

```bash
uv run python "$TRAJMEM_ROOT/scripts/build_history_difference_basis.py" \
  --data "$DATA" \
  --pairs "$TRAJMEM_ROOT/configs/history_pairs.json" \
  --history-config perceptual-framesamp-modul.yaml \
  --count 16 \
  --output "$TRAJMEM_ROOT/results/e12/history_basis.npz"
```

Do not describe random rank-one perturbations as memory hypotheses. They remain a control. The primary E13 result requires a coherent matched-history basis.

## 3. E12 — reward-free clean/degraded-memory recovery

This isolates operator quality without sparse simulator reward. The full-history policy is the self-teacher; the student receives a moderately degraded history. Current images, state, instruction, policy weights, and diffusion noise remain fixed.

```bash
for BASIS in random history hybrid; do
  EXTRA=()
  if [[ "$BASIS" != random ]]; then
    EXTRA=(--history-basis "$TRAJMEM_ROOT/results/e12/history_basis.npz")
  fi
  uv run python "$TRAJMEM_ROOT/scripts/run_e12_operator_basis.py" \
    --checkpoint "$CHECKPOINT" \
    --data "$DATA" \
    --index 0 \
    --preset 16gb \
    --basis "$BASIS" \
    --drop-fraction 0.20 \
    --trust-radius 2.5e-4 \
    --fresh-noises 4 \
    --output "$TRAJMEM_ROOT/results/e12/${BASIS}_state0.json" \
    "${EXTRA[@]}"
done
```

Primary outputs:

- same-noise recovery;
- fresh-noise mean recovery;
- negative-direction and norm-matched-random controls;
- response effective rank;
- applied relative memory norm.

Positive recovery under fresh noises means the edit changed the memory-conditioned policy rather than overfitting one diffusion sample. This is action-behavior recovery, not yet environment success.

## 4. E13 — memory-view branching versus diffusion branching

Use equal total forward-pass budgets. On 16GB the default total is 8 trajectories and evaluates `(B,N)=(1,8),(2,4),(4,2),(8,1)`. On 24GB the default total is 32 and evaluates `(1,32),(4,8),(8,4),(32,1)`.

```bash
uv run python "$TRAJMEM_ROOT/scripts/run_e13_memory_view_branching.py" \
  --checkpoint "$CHECKPOINT" \
  --data "$DATA" \
  --index 0 \
  --preset 16gb \
  --basis "$TRAJMEM_ROOT/results/e12/history_basis.npz" \
  --relative-radius 2.5e-4 \
  --output "$TRAJMEM_ROOT/results/e13/state0_16gb.json"
```

Primary decomposition:

\[
V_\epsilon=\mathbb E_b[\operatorname{Var}_n(A_{b,n})],
\qquad
V_M=\operatorname{Var}_b[\mathbb E_n(A_{b,n})].
\]

The phenomenon of interest is not generic action diversity. It is whether memory-dependent states have substantial `between_memory_variance`, and whether coherent memory branching covers a correct action mode that noise-only branching misses. Target coverage can be added with `--target-actions`; environment success requires the later paired simulator branch experiment.

## 5. What not to claim yet

Do not claim:

- improved RoboMME success before full simulator branches are evaluated;
- that SVD of raw memory is the method;
- that random perturbations are coherent hypotheses;
- that action disagreement alone is calibrated belief uncertainty;
- that the old finite-difference E7 invalidates or validates exact JVP.

The current method claim is narrower: exact JVP measures the action response of structured history-memory directions, and SVD-ridge computes a bounded inverse pullback in that response subspace.
