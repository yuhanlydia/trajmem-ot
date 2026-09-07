#!/usr/bin/env bash
set -euo pipefail
mkdir -p results
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python3 scripts/run_gate0.py --device cuda:0 --seeds 0 2 4 6 --output results/gpu0.json &
pid0=$!
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python3 scripts/run_gate0.py --device cuda:0 --seeds 1 3 5 7 --output results/gpu1.json &
pid1=$!
wait "$pid0" "$pid1"
jq -s 'add' results/gpu0.json results/gpu1.json > results/gate0.json
jq '{n:length, original:(map(.original)|add/length), positive:(map(.positive)|add/length), negative:(map(.negative)|add/length), random:(map(.norm_matched_random)|add/length), improvement_rate:(map(.improvement_rate)|add/length), mean_event_rank:(map(.critical_event_rank)|add/length)}' results/gate0.json
