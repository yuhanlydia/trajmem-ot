# Next GPU Experiments — Corrected Protocol

This handoff supersedes the earlier E13-B/E13-C/E12-S instructions. The old result files are retained, but their aggregate statistics must not be mixed with the corrected protocols below.

## 0. Pull and verify

```bash
cd trajmem-ot
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
mkdir -p results/e13b2/approved results/e13c2 results/e13d results/e12s2 results/e14a
cd "$ROBOMME_DIR"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
```

All commands below run inside the upstream `uv` environment and call scripts from this repository.

## 1. Audit pairs before looking at new outcomes

```bash
uv run python "$TRAJMEM_ROOT/scripts/audit_e13_pairs.py" \
  --checkpoint "$CHECKPOINT" \
  --data "$DATA" \
  --pairs "$TRAJMEM_ROOT/results/e13_pairs.json" \
  --quality-config "$TRAJMEM_ROOT/configs/e13_pair_quality.yaml" \
  --approved-dir "$TRAJMEM_ROOT/results/e13b2/approved" \
  --output "$TRAJMEM_ROOT/results/e13b2/pair_audit_v2.json"
```

This writes `strict.json` and `moderate.json`. Selection uses only prompt match, current image/state similarity, history-mask compatibility, and nontrivial history contrast. Never select pairs using coverage outcomes.

The eight legacy pairs contain few strict examples. Treat them as a smoke set; curate at least 20 strict pairs before a paper-level claim.

## 2. E13-B2 — strict oracle support experiment

```bash
PAIR_FILE="$TRAJMEM_ROOT/results/e13b2/approved/strict.json"
PAIR_COUNT=$(python -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$PAIR_FILE")

for ((PAIR=0; PAIR<PAIR_COUNT; PAIR++)); do
  uv run python "$TRAJMEM_ROOT/scripts/run_e13_oracle_transplant.py" \
    --checkpoint "$CHECKPOINT" \
    --data "$DATA" \
    --pairs "$PAIR_FILE" \
    --pair-index "$PAIR" \
    --require-tier strict \
    --preset 16gb \
    --reference-samples 8 \
    --target-samples 8 \
    --tolerance-multipliers 0.75,1.0,1.25,1.5,2.0 \
    --output "$TRAJMEM_ROOT/results/e13b2/strict_pair_${PAIR}.json"
done

uv run python "$TRAJMEM_ROOT/scripts/analyze_e13_oracle.py" \
  "$TRAJMEM_ROOT"/results/e13b2/strict_pair_*.json \
  --audit "$TRAJMEM_ROOT/results/e13b2/pair_audit_v2.json" \
  --quality-config "$TRAJMEM_ROOT/configs/e13_pair_quality.yaml" \
  --output "$TRAJMEM_ROOT/results/e13b2/strict_summary.json"
```

Primary outputs:

- pair-mean native-history support gain;
- pair-level bootstrap interval and exact sign test;
- gain over the full predeclared tolerance curve;
- headroom recovery;
- history effect normalized by diffusion-noise RMS.

The two transplant directions are averaged within pair. Do not count them as independent samples. This remains an oracle upper bound and does not establish task correctness or success.

## 3. E13-D — can a non-oracle readout recover the missing support?

This starts from the incorrect donor history and generates views only by masking contiguous history spans. The correct memory is used to define an independent target set, but is never inserted into the candidate branches.

```bash
for ((PAIR=0; PAIR<PAIR_COUNT; PAIR++)); do
  uv run python "$TRAJMEM_ROOT/scripts/run_e13_pair_readout_recovery.py" \
    --checkpoint "$CHECKPOINT" \
    --data "$DATA" \
    --pairs "$PAIR_FILE" \
    --pair-index "$PAIR" \
    --preset 16gb \
    --view-counts 1,2,4 \
    --keep-fractions 0.25,0.5,0.75 \
    --primary-view-count 4 \
    --primary-keep-fraction 0.5 \
    --noise-blocks 5 \
    --reference-samples 8 \
    --target-samples 8 \
    --output "$TRAJMEM_ROOT/results/e13d/strict_pair_${PAIR}.json"
done

uv run python "$TRAJMEM_ROOT/scripts/analyze_e13_pair_readout.py" \
  "$TRAJMEM_ROOT"/results/e13d/strict_pair_*.json \
  --output "$TRAJMEM_ROOT/results/e13d/strict_summary.json"
```

Interpretation:

- E13-B2 positive and E13-D positive: simple deployment-safe readout views recover part of the oracle support.
- E13-B2 positive and E13-D null: the gap is supported, but contiguous masks are too coarse; implement a learned low-dimensional FP32 attention-logit bias.
- E13-B2 null on a sufficiently large strict set: re-examine the gap before further method engineering.

## 4. E13-C2 — replicated matched memory/noise decomposition

This is a mechanism experiment, not the primary support test.

```bash
for INDEX in 0 1 2 3 4 5 6 7; do
  uv run python "$TRAJMEM_ROOT/scripts/run_e13_readout_mask_branching.py" \
    --checkpoint "$CHECKPOINT" \
    --data "$DATA" \
    --index "$INDEX" \
    --preset 16gb \
    --keep-fraction 0.50 \
    --view-counts 1,2,4 \
    --noise-blocks 5 \
    --calibrate-native-support \
    --reference-samples 8 \
    --target-samples 8 \
    --output "$TRAJMEM_ROOT/results/e13c2/state_${INDEX}.json"
done

uv run python "$TRAJMEM_ROOT/scripts/analyze_e13_readout.py" \
  "$TRAJMEM_ROOT"/results/e13c2/state_*.json \
  --output "$TRAJMEM_ROOT/results/e13c2/summary.json"
```

The corrected decomposition is

\[
V_{total}=V_{memory\ main}+V_{noise\ main}+V_{interaction}.
\]

The same ordered diffusion seeds are used for every readout view. Report all three components. Do not call the memory-main fraction calibrated belief uncertainty.

## 5. E12-S2 — applied-norm-matched finite-response recovery

Rebuild the history basis on a fresh machine:

```bash
uv run python "$TRAJMEM_ROOT/scripts/build_history_difference_basis.py" \
  --data "$DATA" \
  --pairs "$TRAJMEM_ROOT/results/e13_pairs.json" \
  --history-config perceptual-framesamp-modul.yaml \
  --count 7 \
  --output "$TRAJMEM_ROOT/results/e13_history_basis.npz"
```

Run five corruption seeds on the declared held-out states. The method configuration remains fixed: history basis, probe radius `2.5e-4`, response rank 4, maximum update radius `2.5e-3`.

```bash
for INDEX in 4 5 6 7; do
  for CSEED in 101 202 303 404 505; do
    uv run python "$TRAJMEM_ROOT/scripts/run_e12_secant_recovery.py" \
      --checkpoint "$CHECKPOINT" \
      --data "$DATA" \
      --index "$INDEX" \
      --preset 16gb \
      --basis history \
      --history-basis "$TRAJMEM_ROOT/results/e13_history_basis.npz" \
      --corruption-mode random_rows \
      --corruption-seed "$CSEED" \
      --drop-fraction 0.20 \
      --probe-radii 2.5e-4 \
      --response-rank 4 \
      --max-update-radius 2.5e-3 \
      --fresh-noises 4 \
      --random-controls 16 \
      --control-norm-tolerance 0.03 \
      --output "$TRAJMEM_ROOT/results/e12s2/state_${INDEX}_seed_${CSEED}.json"
  done
done

uv run python "$TRAJMEM_ROOT/scripts/analyze_e12_secant.py" \
  "$TRAJMEM_ROOT"/results/e12s2/state_*.json \
  --heldout-indices 4,5,6,7 \
  --output "$TRAJMEM_ROOT/results/e12s2/heldout_summary.json"
```

The corrected controls are matched after BF16 quantization. The analyzer first averages corruption seeds within state, then bootstraps states. Inspect:

- fresh-noise recovery;
- ours minus median random control;
- negative control;
- maximum applied-norm mismatch;
- actual-versus-linear response cosine/error.

Repeat with `--corruption-mode contiguous_rows` as a harder event-deletion robustness test after the random-row protocol succeeds.

## 6. E14-A — return-tilted OT target fit

Create one manifest per audited branch state following `configs/e14_manifest.example.json`. Every scalar return must correspond to the exact declared `noise_seed` under the same checkpoint, history, current context, and seed protocol.

```bash
uv run python "$TRAJMEM_ROOT/scripts/run_e14_ot_pullback.py" \
  --checkpoint "$CHECKPOINT" \
  --data "$DATA" \
  --manifest "$TRAJMEM_ROOT/results/e14a/state0_manifest.json" \
  --preset 16gb \
  --basis history \
  --history-basis "$TRAJMEM_ROOT/results/e13_history_basis.npz" \
  --probe-radius 2.5e-4 \
  --max-update-radius 2.5e-3 \
  --response-rank 4 \
  --random-controls 8 \
  --output "$TRAJMEM_ROOT/results/e14a/state0.json"
```

Aggregate several states with:

```bash
uv run python "$TRAJMEM_ROOT/scripts/analyze_e14.py" \
  "$TRAJMEM_ROOT"/results/e14a/state*.json \
  --output "$TRAJMEM_ROOT/results/e14a/summary.json"
```

E14-A only establishes open-loop fit to a return-tilted action transport. Save the NPZ and execute `original`, `edited`, `negative`, and random actions from identical simulator branch states before claiming return improvement.

## 7. Decision table

| Result | Interpretation | Next step |
|---|---|---|
| Strict E13-B2 positive | shared-memory support gap survives context controls | evaluate non-oracle views |
| Strict E13-B2 null | old all-pair gain was context-confounded or pair set too small | curate true counterfactual histories |
| E13-D positive | parameter-free readout branching recovers support | closed-loop selection and learning |
| E13-D null but B2 positive | view generator is too weak | learned FP32 attention-logit router |
| Held-out E12-S2 positive vs matched controls | finite-response pullback is robust | connect return/OT |
| E12-S2 actual response poorly matches prediction | secant subspace extrapolates too far | reduce update radius or iterate/relinearize |
| E14-A positive fit but closed-loop null | action-space OT target is not return-causal | redesign target/cost or use iterative feedback |

## Claims still prohibited

Do not claim improved RoboMME success, ground-truth action correctness, calibrated belief uncertainty, or an OT advantage until paired simulator branches establish those endpoints.
