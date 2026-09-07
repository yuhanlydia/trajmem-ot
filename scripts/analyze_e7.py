#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from trajmem_ot.evaluation import select_trust_radius, summarize_memory_line_search


def summarize(rows):
    return summarize_memory_line_search(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = []
    for name in args.inputs:
        rows.extend(json.loads(line) for line in Path(name).read_text().splitlines() if line.strip())
    rows.sort(key=lambda row: row["index"])
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["task"]].append(row)
    losses = np.asarray([-row["q0"] for row in rows])
    threshold = float(np.median(losses))
    report = {
        "metric": "negative demonstration action MSE (offline geometry; not environment return)",
        "n_states": len(rows),
        "random_controls_per_state_radius": len(rows[0]["sweep"][1]["q_random"]),
        "overall": summarize(rows),
        "by_task": {name: summarize(part) for name, part in sorted(grouped.items())},
        "baseline_mse_median": threshold,
        "higher_mse_half": summarize([row for row in rows if -row["q0"] >= threshold]),
        "lower_mse_half": summarize([row for row in rows if -row["q0"] < threshold]),
    }
    report["descriptive_best_rho"] = max(
        report["overall"], key=lambda x: (x["p_plus_gt_minus"], x["p_plus_gt_random"], -x["rho"])
    )["rho"]
    report["selected_rho"] = select_trust_radius(report["overall"])
    report["gate_pass"] = report["selected_rho"] is not None
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
