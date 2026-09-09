#!/usr/bin/env python3
"""Aggregate E12-S/S2 finite-response recovery at the state level."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from trajmem_ot.stats import exact_two_sided_sign_test, percentile_bootstrap


def _parse_int_list(value: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("indices must be non-negative")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument(
        "--heldout-indices",
        type=_parse_int_list,
        default=(),
        help="optional comma-separated held-out state indices",
    )
    return parser.parse_args()


def _row_random_fresh(row: dict[str, Any]) -> float:
    if "fresh_noise_random_median_recovery" in row:
        return float(row["fresh_noise_random_median_recovery"])
    return float(row.get("random_direction_recovery", 0.0))


def _row_negative_fresh(row: dict[str, Any]) -> float:
    if "fresh_noise_negative_mean_recovery" in row:
        return float(row["fresh_noise_negative_mean_recovery"])
    return float(row.get("negative_direction_recovery", 0.0))


def _state_means(
    records: Iterable[tuple[int, int, dict[str, Any]]]
) -> list[dict[str, float | int]]:
    by_state: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for state_index, corruption_seed, row in records:
        by_state[state_index].append((corruption_seed, row))

    values: list[dict[str, float | int]] = []
    for state_index, rows in sorted(by_state.items()):
        ours = np.asarray(
            [float(row["fresh_noise_mean_recovery"]) for _, row in rows],
            dtype=np.float64,
        )
        random = np.asarray([_row_random_fresh(row) for _, row in rows])
        negative = np.asarray([_row_negative_fresh(row) for _, row in rows])
        same = np.asarray([float(row["same_noise_recovery"]) for _, row in rows])
        norm_error = np.asarray(
            [
                max(
                    [
                        float(control.get("relative_norm_error", 0.0))
                        for control in row.get("random_controls", [])
                    ]
                    + [
                        float(
                            row.get("negative_control", {}).get(
                                "relative_norm_error", 0.0
                            )
                        )
                    ]
                )
                for _, row in rows
            ],
            dtype=np.float64,
        )
        fidelity = np.asarray(
            [
                float(
                    row.get("response_fidelity", {}).get(
                        "actual_vs_linear_cosine", np.nan
                    )
                )
                for _, row in rows
            ],
            dtype=np.float64,
        )
        values.append(
            {
                "state_index": state_index,
                "n_corruptions": len(rows),
                "mean_fresh_recovery": float(ours.mean()),
                "mean_random_recovery": float(random.mean()),
                "mean_negative_recovery": float(negative.mean()),
                "mean_same_noise_recovery": float(same.mean()),
                "mean_ours_minus_random": float((ours - random).mean()),
                "max_control_norm_error": float(norm_error.max()),
                "mean_response_fidelity_cosine": (
                    float(np.nanmean(fidelity))
                    if np.any(np.isfinite(fidelity))
                    else None
                ),
            }
        )
    return values


def _summary(
    state_rows: list[dict[str, float | int]],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any] | None:
    if not state_rows:
        return None
    ours = np.asarray([row["mean_fresh_recovery"] for row in state_rows])
    random = np.asarray([row["mean_random_recovery"] for row in state_rows])
    negative = np.asarray([row["mean_negative_recovery"] for row in state_rows])
    effect = ours - random
    same = np.asarray([row["mean_same_noise_recovery"] for row in state_rows])
    fidelity = np.asarray(
        [
            np.nan
            if row["mean_response_fidelity_cosine"] is None
            else float(row["mean_response_fidelity_cosine"])
            for row in state_rows
        ],
        dtype=np.float64,
    )
    return {
        "n_states": len(state_rows),
        "n_corruptions": int(sum(int(row["n_corruptions"]) for row in state_rows)),
        "state_indices": [int(row["state_index"]) for row in state_rows],
        "median_fresh_noise_recovery": float(np.median(ours)),
        "mean_fresh_noise_recovery_ci": asdict(
            percentile_bootstrap(ours, resamples=resamples, seed=seed)
        ),
        "fraction_positive_fresh_noise_recovery": float(np.mean(ours > 0)),
        "median_random_control_recovery": float(np.median(random)),
        "median_negative_control_recovery": float(np.median(negative)),
        "median_same_noise_recovery": float(np.median(same)),
        "median_ours_minus_random": float(np.median(effect)),
        "mean_ours_minus_random_ci": asdict(
            percentile_bootstrap(effect, resamples=resamples, seed=seed + 1)
        ),
        "ours_minus_random_sign_test": exact_two_sided_sign_test(effect),
        "maximum_control_norm_relative_error": float(
            max(float(row["max_control_norm_error"]) for row in state_rows)
        ),
        "median_response_fidelity_cosine": (
            float(np.nanmedian(fidelity))
            if np.any(np.isfinite(fidelity))
            else None
        ),
        "state_level": state_rows,
    }


def main() -> None:
    args = parse_args()
    reports = []
    for path in args.inputs:
        report = json.loads(path.read_text())
        if report.get("experiment") in {
            "E12S_deployed_secant_memory_recovery",
            "E12S2_deployed_secant_memory_recovery",
        }:
            reports.append(report)
    if not reports:
        raise ValueError("no E12-S or E12-S2 reports were provided")

    grouped: dict[tuple[str, float, str], list[tuple[int, int, dict]]] = defaultdict(list)
    for report in reports:
        state_index = int(report["index"])
        corruption_seed = int(report.get("corruption_seed", report.get("seed", state_index)))
        corruption_mode = str(report.get("corruption_mode", "legacy_random_rows"))
        for row in report["radius_results"]:
            key = (report["basis"], float(row["probe_radius"]), corruption_mode)
            grouped[key].append((state_index, corruption_seed, row))

    heldout = set(args.heldout_indices)
    groups = []
    for offset, ((basis, radius, mode), records) in enumerate(sorted(grouped.items())):
        state_rows = _state_means(records)
        all_summary = _summary(
            state_rows,
            resamples=args.bootstrap_resamples,
            seed=args.seed + 100 * offset,
        )
        heldout_rows = [row for row in state_rows if int(row["state_index"]) in heldout]
        development_rows = [
            row for row in state_rows if int(row["state_index"]) not in heldout
        ]
        groups.append(
            {
                "basis": basis,
                "probe_radius": radius,
                "corruption_mode": mode,
                "all": all_summary,
                "development": (
                    _summary(
                        development_rows,
                        resamples=args.bootstrap_resamples,
                        seed=args.seed + 100 * offset + 10,
                    )
                    if heldout
                    else None
                ),
                "heldout": (
                    _summary(
                        heldout_rows,
                        resamples=args.bootstrap_resamples,
                        seed=args.seed + 100 * offset + 20,
                    )
                    if heldout
                    else None
                ),
            }
        )

    output = {
        "experiment": "E12S2_deployed_secant_memory_recovery_aggregate",
        "n_input_reports": len(reports),
        "heldout_indices": sorted(heldout),
        "groups": groups,
        "inference_unit": (
            "corruption seeds are averaged within state; bootstrap and sign tests "
            "resample states, not individual noise draws"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
