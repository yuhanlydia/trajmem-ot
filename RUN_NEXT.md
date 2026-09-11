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

## Remaining integrations

Synthetic row corruption establishes the GPU path and mechanism only. Native
interference requires actual relevant/distractor history concatenation and a
source-index manifest. Persistence requires later query observations with teacher
retrieval removed from the repaired branch. Physical evaluation requires correct
action unnormalization and alignment to the same simulator state. These stages
need executed evidence before being described as completed.
