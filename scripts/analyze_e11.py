#!/usr/bin/env python3
"""Aggregate one or more E11 exact-JVP JSON reports."""

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
    rows = [json.loads(path.read_text()) for path in args.inputs]
    comparisons = [entry for row in rows for entry in row.get("fd_comparisons", [])]
    report = {
        "experiment": "E11_exact_jvp_aggregate",
        "n_states": len(rows),
        "presets": sorted({row.get("preset") for row in rows}),
        "median_warm_latency_s": float(np.median([row["warm_call_latency_s"] for row in rows])),
        "median_effective_rank_90": float(np.median([row["effective_rank_90"] for row in rows])),
        "min_fd_cosine": min((row["cosine"] for row in comparisons), default=None),
        "median_fd_cosine": float(np.median([row["cosine"] for row in comparisons])) if comparisons else None,
        "max_fd_relative_error": max((row["relative_error"] for row in comparisons), default=None),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
