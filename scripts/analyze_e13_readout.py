#!/usr/bin/env python3
"""Aggregate E13-C2 matched history-readout branching reports."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from trajmem_ot.stats import percentile_bootstrap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260909)
    return parser.parse_args()


def _bootstrap(values: list[float], args: argparse.Namespace, offset: int) -> dict | None:
    if not values:
        return None
    return asdict(
        percentile_bootstrap(
            np.asarray(values),
            resamples=args.bootstrap_resamples,
            seed=args.seed + offset,
        )
    )


def main() -> None:
    args = parse_args()
    reports = [json.loads(path.read_text()) for path in args.inputs]
    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for report in reports:
        for row in report["allocations"]:
            grouped[(int(row["memory_views"]), int(row["noises_per_view"]))].append(row)

    summaries = []
    for offset, ((views, noises), values) in enumerate(sorted(grouped.items())):
        robot = [row["robot_action_8d"] for row in values]
        matched_available = all("matched_memory_main_fraction" in row for row in robot)
        memory_fractions = [
            float(
                row[
                    "matched_memory_main_fraction"
                    if "matched_memory_main_fraction" in row
                    else "memory_variance_fraction"
                ]
            )
            for row in robot
        ]
        noise_fractions = [
            float(row.get("matched_noise_main_fraction", 1.0 - memory_fractions[i]))
            for i, row in enumerate(robot)
        ]
        interaction_fractions = [
            float(row.get("matched_interaction_fraction", 0.0)) for row in robot
        ]
        coverage_gains = []
        for row in values:
            native = row.get("native_support")
            if native is None:
                continue
            if "mean_coverage_gain_vs_noise_only" in native:
                coverage_gains.append(float(native["mean_coverage_gain_vs_noise_only"]))
            elif "coverage_gain_vs_noise_only" in native:
                coverage_gains.append(float(native["coverage_gain_vs_noise_only"]))
        summaries.append(
            {
                "memory_views": views,
                "noises_per_view": noises,
                "n_states": len(values),
                "decomposition": "matched_two_way" if matched_available else "legacy_one_way",
                "median_memory_main_fraction": float(np.median(memory_fractions)),
                "memory_main_fraction_mean_ci": _bootstrap(
                    memory_fractions, args, 10 * offset
                ),
                "median_noise_main_fraction": float(np.median(noise_fractions)),
                "median_interaction_fraction": float(np.median(interaction_fractions)),
                "median_memory_main_variance": float(
                    np.median(
                        [
                            row.get(
                                "matched_memory_main_variance",
                                row.get("between_memory_variance", 0.0),
                            )
                            for row in robot
                        ]
                    )
                ),
                "median_native_support_coverage_gain": (
                    float(np.median(coverage_gains)) if coverage_gains else None
                ),
                "native_support_coverage_gain_mean_ci": _bootstrap(
                    coverage_gains, args, 10 * offset + 1
                ),
            }
        )

    output = {
        "experiment": "E13C2_history_readout_mask_branching_aggregate",
        "n_states": len(reports),
        "allocations": summaries,
        "interpretation": (
            "matched_memory_main_fraction is the balanced view main effect under "
            "seed-matched noise blocks; interaction is reported separately. It is "
            "an action-separation statistic, not calibrated belief uncertainty."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
