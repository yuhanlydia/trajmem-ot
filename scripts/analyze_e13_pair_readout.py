#!/usr/bin/env python3
"""Aggregate E13-D pair-readout recovery with pair-level inference."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from trajmem_ot.stats import exact_two_sided_sign_test, percentile_bootstrap


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260909)
    return parser.parse_args()


def main():
    args = parse_args()
    reports = []
    for path in args.inputs:
        report = json.loads(path.read_text())
        if report.get("experiment") == "E13D_pair_readout_recovery":
            reports.append(report)
    if not reports:
        raise ValueError("no E13-D reports were provided")
    gains = np.asarray(
        [float(row["pair_mean_primary_readout_coverage_gain"]) for row in reports]
    )
    output = {
        "experiment": "E13D_pair_readout_recovery_aggregate",
        "n_pairs": len(reports),
        "pair_indices": [int(row["source_pair_index"]) for row in reports],
        "mean_pair_primary_coverage_gain_ci": asdict(
            percentile_bootstrap(
                gains, resamples=args.bootstrap_resamples, seed=args.seed
            )
        ),
        "median_pair_primary_coverage_gain": float(np.median(gains)),
        "fraction_positive_pairs": float(np.mean(gains > 0)),
        "pair_sign_test": exact_two_sided_sign_test(gains),
        "inference_unit": "pairs; the two transplant directions are averaged within pair",
        "claim_scope": (
            "native-history support recovery under a preregistered readout "
            "configuration, not per-pair best selection and not task success"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
