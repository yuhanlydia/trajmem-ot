# FRMD experiment runbook

FRMD is the active direction. The fetched upstream main did not contain the
announced E14-R implementation; the repair modules here are a new implementation
of the supplied finite-response protocol. Historical commands and results remain
available in [the legacy runbook](docs/LEGACY_RUN_NEXT_2026-09-11.md).

## Research continuation

All experiment stages use an exploratory continuation policy, per the updated
research instruction on 2026-09-11. There is no required accuracy, success gain,
repair ratio, p-value, task count, or persistence threshold that automatically
stops subsequent experiments. Assess positive behavior and reusability across
histories, tasks, and memory mechanisms; retain neutral and negative results.

Held-out per-edit acceptance and rollback are part of the repair algorithm and
remain enabled. Controls, real data provenance, and separation of optimization,
validation, and evaluation noises remain necessary to interpret the evidence.
An action-space improvement is reported as such; it is not physical success.

## Verify and inspect the budget

```bash
python -m pip install -e '.[dev,jax]'
pytest -q
python -m compileall -q src scripts tests
python scripts/run_frmd_smoke.py --help
python scripts/run_frmd_smoke.py --config configs/frmd_smoke_16gb.yaml --dry-run
```

The default per-case budget is 256 two-sided probe trajectory queries, 340 total
repair queries, and 40 additional control queries. The nominal 16GB preset
specifies query settings; measure deployment VRAM for the selected runtime.

## Real checkpoint smoke

Install the pinned runtime using `scripts/bootstrap_robomme.sh`. The upstream
configuration loader resolves paths relative to its working directory:

```bash
export TRAJMEM_ROOT="$(pwd)"
export ROBOMME_DIR="$TRAJMEM_ROOT/third_party/robomme_policy_learning"
export PYTHONPATH="$TRAJMEM_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Set CHECKPOINT to the checkpoint step directory and DATA to preprocessed data.
cd "$ROBOMME_DIR"
.venv/bin/python "$TRAJMEM_ROOT/scripts/run_frmd_smoke.py" \
  --checkpoint "$CHECKPOINT" --data "$DATA" --index 0 \
  --config "$TRAJMEM_ROOT/configs/frmd_smoke_16gb.yaml" --corruption random_rows \
  --output "$TRAJMEM_ROOT/results/frmd/smoke_random.json"
.venv/bin/python "$TRAJMEM_ROOT/scripts/run_frmd_smoke.py" \
  --checkpoint "$CHECKPOINT" --data "$DATA" --index 0 \
  --config "$TRAJMEM_ROOT/configs/frmd_smoke_16gb.yaml" --corruption contiguous_rows \
  --output "$TRAJMEM_ROOT/results/frmd/smoke_contiguous.json"
```

Use `--iterations 1` for the one-step ablation and `--target centroid` or
`--target ot` for target ablations, each with a new output path. The default basis
is independent random rank-one directions. An explicit `--history-basis` NPZ
must contain exactly K directions and its content hash is recorded. No oracle
corruption-reversal direction is inserted silently.

JSON reports contain rollback traces, actual query counts, fresh-noise recovery,
and negative/random/direct-interpolation controls matched to the applied memory
norm. NPZ artifacts contain memory tensors and normalized actions, not simulator
commands. Preserve artifacts locally; publish lightweight reports and provenance.

## Native development pilot and offline persistence

The native tokendrop integration is implemented and has executed real-history
cases. Prepare a completed per-episode catalog with actual raw files and indices,
then build a fresh manifest with paths on your machine:

```bash
python scripts/build_e14r_interference_manifest.py \
  --catalog "$CATALOG" --data "$DATA" --output "$MANIFEST"
python scripts/run_e14r_pilot.py --manifest "$MANIFEST" \
  --config configs/frmd_smoke_16gb.yaml --dry-run
cd "$ROBOMME_DIR"
.venv/bin/python "$TRAJMEM_ROOT/scripts/run_e14r_pilot.py" \
  --manifest "$MANIFEST" --checkpoint "$CHECKPOINT" \
  --config "$TRAJMEM_ROOT/configs/frmd_smoke_16gb.yaml" \
  --output-dir "$NATIVE_RESULTS" --resume
```

This runner defaults to the available retrieved-history difference plus seven
random directions. Use `--basis random` for the independent-basis ablation and
`--iterations 1`, `--target centroid`, or `--target ot` for method ablations, with
separate output directories. Resume refuses incompatible source/configuration
records. There are 256 probe queries and 390 base queries including calibration
and controls; offline persistence adds 24 queries per available horizon.

Persistence transfers the applied edit only to surviving raw-source tokens after
causal buffer updates, without teacher access on the repaired branch. Teacher
queries are used for offline scoring. Recorded future observations do not measure
closed-loop interaction. Missing horizons are recorded; no raw indices are invented.

## Headless physical chunks and descriptive analysis

Use the benchmark environment, with the same PYTHONPATH, for saved unnormalized
action chunks. The entrypoint verifies raw seed alignment and identical initial
fingerprints for all six conditions; `--all-noises` evaluates all held-out replicas.

```bash
"$ROBOMME_DIR/third_party/robomme_benchmark/.venv/bin/python" \
  "$TRAJMEM_ROOT/scripts/run_e14r_one_chunk.py" \
  --report "$NATIVE_RESULTS/PatternLock-episode-0-k3.json" \
  --all-noises --output "$PHYSICAL_RESULT"
python "$TRAJMEM_ROOT/scripts/analyze_e14r.py" \
  --repairs "$NATIVE_RESULTS/PatternLock-episode-0-k3.json" \
  --physical "$PHYSICAL_RESULT" --output "$ANALYSIS"
```

The analyzer accepts multiple repair/physical reports from one full configuration,
including gzip JSON. It averages noise replicas within each case and validates
physical report provenance. Ongoing chunks remain distinct from terminal failure.
No metric threshold automatically stops experimentation. Current evidence and
limitations are in [the dated status](results/frmd/2026-09-11-status.md).
