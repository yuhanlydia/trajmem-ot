#!/usr/bin/env python3
"""Aggregate E12-S deployed finite-response recovery reports."""

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


def _median(values):
    return float(np.median(values)) if values else None


def main() -> None:
    args = parse_args()
    reports = [json.loads(path.read_text()) for path in args.inputs]
    grouped: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for report in reports:
        for row in report["radius_results"]:
            grouped[(report["basis"], float(row["probe_radius"]))].append(row)

    rows = []
    for (basis, radius), values in sorted(grouped.items()):
        fresh = [float(row["fresh_noise_mean_recovery"]) for row in values]
        same = [float(row["same_noise_recovery"]) for row in values]
        negative = [float(row["negative_direction_recovery"]) for row in values]
        random = [float(row["random_direction_recovery"]) for row in values]
        residual_ratio = [
            float(row["linear_target_residual_after"])
            / max(float(row["linear_target_residual_before"]), 1e-12)
            for row in values
        ]
        rows.append(
            {
                "basis": basis,
                "probe_radius": radius,
                "n_states": len(values),
                "median_fresh_noise_recovery": _median(fresh),
                "fraction_positive_fresh_noise_recovery": float(
                    np.mean(np.asarray(fresh) > 0)
                ),
                "median_same_noise_recovery": _median(same),
                "median_negative_direction_recovery": _median(negative),
                "median_random_direction_recovery": _median(random),
                "median_fresh_minus_random": _median(
                    [left - right for left, right in zip(fresh, random, strict=True)]
                ),
                "median_linear_residual_ratio": _median(residual_ratio),
                "median_applied_relative_delta_norm": _median(
                    [float(row["applied_relative_delta_norm"]) for row in values]
                ),
            }
        )

    report = {
        "experiment": "E12S_deployed_secant_memory_recovery_aggregate",
        "n_input_reports": len(reports),
        "groups": rows,
        "selection_note": (
            "choose basis/radius/damping on a predeclared development split; "
            "report held-out states without retuning"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
