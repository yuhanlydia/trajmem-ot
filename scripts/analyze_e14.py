#!/usr/bin/env python3
"""Aggregate E14-A open-loop OT pullback reports at the state level."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from trajmem_ot.stats import exact_two_sided_sign_test, percentile_bootstrap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260909)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = []
    for path in args.inputs:
        report = json.loads(path.read_text())
        if report.get("experiment") == "E14A_return_tilted_ot_finite_response_pullback":
            reports.append(report)
    if not reports:
        raise ValueError("no E14-A reports were provided")
    fit = np.asarray([float(row["actual_target_fit"]) for row in reports])
    random = np.asarray(
        [
            float(np.median([control["target_fit"] for control in row["random_controls"]]))
            for row in reports
        ]
    )
    negative = np.asarray(
        [float(row["negative_control"]["target_fit"]) for row in reports]
    )
    effect = fit - random
    output = {
        "experiment": "E14A_return_tilted_ot_finite_response_pullback_aggregate",
        "n_states": len(reports),
        "state_indices": [int(row["index"]) for row in reports],
        "median_actual_target_fit": float(np.median(fit)),
        "mean_actual_target_fit_ci": asdict(
            percentile_bootstrap(
                fit, resamples=args.bootstrap_resamples, seed=args.seed
            )
        ),
        "median_random_target_fit": float(np.median(random)),
        "median_negative_target_fit": float(np.median(negative)),
        "median_ours_minus_random": float(np.median(effect)),
        "mean_ours_minus_random_ci": asdict(
            percentile_bootstrap(
                effect, resamples=args.bootstrap_resamples, seed=args.seed + 1
            )
        ),
        "ours_minus_random_sign_test": exact_two_sided_sign_test(effect),
        "claim_scope": (
            "open-loop action-transport fit only; paired simulator returns and success "
            "must be analyzed separately"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
