#!/usr/bin/env python3
"""Aggregate E13-C history-readout branching reports."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = [json.loads(path.read_text()) for path in args.inputs]
    grouped = defaultdict(list)
    for report in reports:
        for row in report["allocations"]:
            key = (int(row["memory_views"]), int(row["noises_per_view"]))
            grouped[key].append(row)

    summaries = []
    for (views, noises), values in sorted(grouped.items()):
        robot = [row["robot_action_8d"] for row in values]
        summaries.append(
            {
                "memory_views": views,
                "noises_per_view": noises,
                "n_states": len(values),
                "median_between_memory_variance": float(
                    np.median([row["between_memory_variance"] for row in robot])
                ),
                "median_within_noise_variance": float(
                    np.median([row["within_noise_variance"] for row in robot])
                ),
                "median_memory_variance_fraction": float(
                    np.median([row["memory_variance_fraction"] for row in robot])
                ),
                "coverage_fraction_when_available": (
                    float(
                        np.mean(
                            [
                                row["target_coverage"]["covered"]
                                for row in values
                                if "target_coverage" in row
                            ]
                        )
                    )
                    if any("target_coverage" in row for row in values)
                    else None
                ),
            }
        )

    output = {
        "experiment": "E13C_history_readout_mask_branching_aggregate",
        "n_states": len(reports),
        "allocations": summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
