# Next GPU Experiments

This file is the current execution handoff. It supersedes the earlier plan that sent the released BF16 policy directly into JVP-based E12.

## Why the protocol changed

The FP32 shadow result shows that the JVP wiring is correct on a numerically smooth path, while the deployed BF16 modulation/LLM path is not faithfully described by its infinitesimal AD tangent. Therefore:

1. use FP32 JVP only for mechanism diagnostics;
2. use full forward transplants to test the research phenomenon;
3. use boolean readout masks as the first deployment-safe view generator;
4. use finite BF16 action responses for any deployed inverse pullback.

Do **not** run `scripts/run_e12_operator_basis.py` as the main next experiment.

## 0. Pull and verify

```bash
git pull --ff-only
python -m pip install -e '.[dev,jax]'
pytest -q
python -m compileall -q src scripts tests
```

Prepare the released runtime:

```bash
export TRAJMEM_ROOT="$(pwd)"
export ROBOMME_DIR="$TRAJMEM_ROOT/third_party/robomme_policy_learning"
export CHECKPOINT=/absolute/path/to/perceptual-framesamp-modul/79999
export DATA=/absolute/path/to/robomme_preprocessed_data_sample

bash scripts/bootstrap_robomme.sh
mkdir -p results/e11d results/e12s results/e13b results/e13c
cd "$ROBOMME_DIR"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
```

All commands below execute from the upstream `uv` environment and call scripts in this repository.

## 1. E11-D — confirm the FP32 root-cause diagnosis

Start with states 0 and 1:

```bash
for INDEX in 0 1; do
  for STEPS in 1 10; do
    JAX_DEFAULT_MATMUL_PRECISION=highest \
    uv run python "$TRAJMEM_ROOT/scripts/run_e11_robomme_jvp.py" \
      --checkpoint "$CHECKPOINT" \
      --data "$DATA" \
      --index "$INDEX" \
      --preset 16gb \
      --num-steps "$STEPS" \
      --fd-directions 2 \
      --fd-radii 2.5e-4 \
      --fp32-shadow \
      --output "$TRAJMEM_ROOT/results/e11d/state_${INDEX}_steps_${STEPS}.json"
  done
done
```

Then extend to states 0–7 at one and ten flow steps. Record separately:

- `base_primal_delta_norm`;
- `cosine_robot_8d`;
- `relative_error`;
- response singular values and `effective_rank_90`.

This experiment validates the smooth diagnostic operator. It does not convert the released checkpoint into an FP32 deployment model.

## 2. E13-B — oracle full-history transplant

This is the first decisive test of the research gap. It uses complete history bundles rather than a `0.025%` perturbation.

The committed `results/e13_pairs.json` contains candidate same-task pairs. Inspect the reported image/state distances and retain only pairs whose histories represent a plausible competing interpretation for the fixed current context.

### 16GB

```bash
for PAIR in 0 1 2 3 4 5 6 7; do
  uv run python "$TRAJMEM_ROOT/scripts/run_e13_oracle_transplant.py" \
    --checkpoint "$CHECKPOINT" \
    --data "$DATA" \
    --pairs "$TRAJMEM_ROOT/results/e13_pairs.json" \
    --pair-index "$PAIR" \
    --preset 16gb \
    --reference-samples 4 \
    --target-samples 4 \
    --output "$TRAJMEM_ROOT/results/e13b/pair_${PAIR}.json"
done

uv run python "$TRAJMEM_ROOT/scripts/analyze_e13_oracle.py" \
  "$TRAJMEM_ROOT"/results/e13b/pair_*.json \
  --output "$TRAJMEM_ROOT/results/e13b/summary.json"
```

For each direction, the comparison is:

\[
(1\text{ wrong shared memory},8\text{ noises})
\quad\text{vs}\quad
(\text{wrong}+\text{correct memory},4\text{ noises each}).
\]

Primary outputs:

- calibrated correct-mode coverage;
- coverage gain from adding the correct memory hypothesis;
- nearest-distance improvement;
- paired action difference under identical noise;
- prompt/image/state pair diagnostics.

A positive oracle coverage gain shows that the missing action support can be recovered by changing the history hypothesis rather than drawing more noise from the wrong shared memory.

## 3. E13-C — deployment-faithful readout-mask branching

This experiment leaves all memory values unchanged and changes only the valid history span exposed to memory attention.

```bash
for INDEX in 0 1 2 3 4 5 6 7; do
  uv run python "$TRAJMEM_ROOT/scripts/run_e13_readout_mask_branching.py" \
    --checkpoint "$CHECKPOINT" \
    --data "$DATA" \
    --index "$INDEX" \
    --preset 16gb \
    --keep-fraction 0.50 \
    --view-counts 1,2,4 \
    --output "$TRAJMEM_ROOT/results/e13c/state_${INDEX}.json"
done
```

Matched allocations are `(1,8)`, `(2,4)`, and `(4,2)`. Do not interpret `(8,1)` as a variance comparison.

Primary outputs:

- robot-8D between-memory variance;
- within-view diffusion variance;
- target-mode coverage when a target action file is supplied;
- valid token count per readout view.

If oracle transplants help but contiguous masks do not, the gap is real but the view generator is too weak. The next model should then learn a low-dimensional FP32 attention-logit bias.

## 4. E12-S — deployed BF16 finite-response recovery

Run this only after E13-B confirms that alternative history conditioning can recover a useful mode. This path never consumes the unvalidated BF16 AD tangent.

First build or reuse a coherent history basis, then run one-state diagnostics:

```bash
for BASIS in random history hybrid; do
  EXTRA=()
  if [[ "$BASIS" != random ]]; then
    EXTRA=(--history-basis "$TRAJMEM_ROOT/results/e13_history_basis.npz")
  fi
  uv run python "$TRAJMEM_ROOT/scripts/run_e12_secant_recovery.py" \
    --checkpoint "$CHECKPOINT" \
    --data "$DATA" \
    --index 0 \
    --preset 16gb \
    --basis "$BASIS" \
    --drop-fraction 0.20 \
    --probe-radii 2.5e-4,1e-3,2.5e-3,5e-3 \
    --max-update-radius 2.5e-3 \
    --fresh-noises 4 \
    --output "$TRAJMEM_ROOT/results/e12s/${BASIS}_state0.json" \
    "${EXTRA[@]}"
done
```

Primary outputs:

- same-noise recovery;
- fresh-noise recovery;
- negative and norm-matched random controls;
- actual quantized edit norm;
- finite response singular values;
- linear prediction residual versus actual recovery.

A useful deployed response model should improve fresh-noise recovery, not only fit the probe noise.

## 5. E14 — trajectory OT in response/readout space

Only after a useful E12-S or learned readout operator is established:

\[
\{\tau_i,R_i\}
\rightarrow
u^{OT}
\rightarrow
\Delta z\text{ or }\Delta M_{sec}.
\]

Compare:

- best-of-N;
- return-weighted centroid;
- OT transport;
- random response-space edit;
- negative transport direction.

The primary endpoint is paired simulator success and subgoal progress under fresh diffusion noises.

## Interpretation matrix

| Result | Meaning | Next action |
|---|---|---|
| E13-B positive | shared-memory support gap exists | improve automatic view generation |
| E13-B null | current pair construction or core gap is unsupported | curate true counterfactual histories before method work |
| E13-B positive, E13-C null | simple masks are too weak | implement learned FP32 attention-logit bias |
| E12-S positive | deployed finite response is a usable pullback operator | connect return/OT |
| E12-S null | raw memory is not a useful deployed control surface | optimize only the readout variable |

## Claims that remain prohibited

Do not claim improved RoboMME success, calibrated belief uncertainty, or a learned memory-hypothesis posterior until closed-loop paired simulator experiments are complete.
