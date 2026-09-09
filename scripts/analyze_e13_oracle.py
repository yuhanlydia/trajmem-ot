#!/usr/bin/env python3
"""Aggregate E13-B oracle-transplant reports."""

from __future__ import annotations

import argparse
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
    reports = [
        report
        for path in args.inputs
        if (report := json.loads(path.read_text())).get("experiment")
        == "E13B_oracle_history_transplant"
    ]
    if not reports:
        raise ValueError("no E13-B oracle history-transplant reports were provided")
    directions = [direction for report in reports for direction in report["directions"]]
    gains = [direction["coverage"]["coverage_gain"] for direction in directions]
    improvements = [
        direction["coverage"]["mean_distance_improvement"] for direction in directions
    ]
    paired = [direction["paired_memory_effect_mean"] for direction in directions]
    output = {
        "experiment": "E13B_oracle_history_transplant_aggregate",
        "n_pairs": len(reports),
        "n_directions": len(directions),
        "mean_coverage_gain": float(np.mean(gains)),
        "median_coverage_gain": float(np.median(gains)),
        "fraction_positive_coverage_gain": float(np.mean(np.asarray(gains) > 0)),
        "mean_distance_improvement": float(np.mean(improvements)),
        "median_paired_memory_effect": float(np.median(paired)),
        "prompt_match_fraction": float(
            np.mean([report["context_diagnostics"]["prompt_match"] for report in reports])
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
