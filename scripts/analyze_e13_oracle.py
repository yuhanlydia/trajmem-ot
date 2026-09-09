#!/usr/bin/env python3
"""Pair-level aggregate analysis for E13-B/B2 oracle history transplants."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np

from trajmem_ot.pair_quality import assess_pair_quality, load_pair_quality_tiers
from trajmem_ot.stats import exact_two_sided_sign_test, percentile_bootstrap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--quality-config", type=Path)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260909)
    return parser.parse_args()


def _audit_lookup(path: Path | None) -> dict[int, dict[str, Any]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text())
    return {int(row["pair_index"]): row for row in payload.get("pairs", [])}


def _pair_quality(
    report: dict[str, Any],
    *,
    audit: dict[int, dict[str, Any]],
    tiers: dict[str, Any],
) -> dict[str, bool]:
    source_index = int(report.get("source_pair_index", report["pair_index"]))
    row = audit.get(source_index)
    if row is None:
        return {name: False for name in tiers}
    embedded = row.get("quality_tiers", {})
    result: dict[str, bool] = {}
    for name, criteria in tiers.items():
        if name in embedded:
            result[name] = bool(embedded[name]["eligible"])
        else:
            result[name] = assess_pair_quality(row, criteria).eligible
    return result


def _direction_gain(direction: dict[str, Any]) -> float:
    return float(direction["coverage"]["coverage_gain"])


def _pair_row(report: dict[str, Any]) -> dict[str, Any]:
    directions = report["directions"]
    gains = np.asarray([_direction_gain(row) for row in directions], dtype=np.float64)
    improvements = np.asarray(
        [float(row["coverage"]["mean_distance_improvement"]) for row in directions]
    )
    headrooms = [
        row.get("native_support_headroom_recovery")
        for row in directions
        if row.get("native_support_headroom_recovery") is not None
    ]
    normalized_effect = [
        float(row.get("paired_memory_effect_to_noise_rms", np.nan))
        for row in directions
    ]
    curve_by_multiplier: dict[str, list[float]] = {}
    for direction in directions:
        for point in direction.get("coverage_curve", []):
            key = f"{float(point['tolerance_multiplier']):g}"
            curve_by_multiplier.setdefault(key, []).append(
                float(point["coverage"]["coverage_gain"])
            )
    return {
        "pair_index": int(report["pair_index"]),
        "source_pair_index": int(report.get("source_pair_index", report["pair_index"])),
        "pair_mean_coverage_gain": float(gains.mean()),
        "pair_min_coverage_gain": float(gains.min()),
        "pair_mean_distance_improvement": float(improvements.mean()),
        "pair_mean_headroom_recovery": (
            float(np.mean(headrooms)) if headrooms else None
        ),
        "pair_median_memory_effect_to_noise_rms": (
            float(np.nanmedian(normalized_effect))
            if np.any(np.isfinite(normalized_effect))
            else None
        ),
        "direction_gains": gains.tolist(),
        "coverage_curve_pair_mean_gain": {
            key: float(np.mean(values)) for key, values in sorted(curve_by_multiplier.items())
        },
        "context": report.get("context_diagnostics", {}),
    }


def _summarize(
    rows: list[dict[str, Any]],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any] | None:
    if not rows:
        return None
    gains = np.asarray([row["pair_mean_coverage_gain"] for row in rows])
    improvements = np.asarray([row["pair_mean_distance_improvement"] for row in rows])
    headrooms = np.asarray(
        [
            row["pair_mean_headroom_recovery"]
            for row in rows
            if row["pair_mean_headroom_recovery"] is not None
        ],
        dtype=np.float64,
    )
    normalized = np.asarray(
        [
            row["pair_median_memory_effect_to_noise_rms"]
            for row in rows
            if row["pair_median_memory_effect_to_noise_rms"] is not None
        ],
        dtype=np.float64,
    )
    multipliers = sorted(
        {
            key
            for row in rows
            for key in row["coverage_curve_pair_mean_gain"]
        },
        key=float,
    )
    curve = []
    for offset, multiplier in enumerate(multipliers):
        values = np.asarray(
            [
                row["coverage_curve_pair_mean_gain"][multiplier]
                for row in rows
                if multiplier in row["coverage_curve_pair_mean_gain"]
            ]
        )
        curve.append(
            {
                "tolerance_multiplier": float(multiplier),
                "n_pairs": int(values.size),
                "mean_pair_gain_ci": asdict(
                    percentile_bootstrap(
                        values,
                        resamples=resamples,
                        seed=seed + 100 + offset,
                    )
                ),
                "fraction_positive_pairs": float(np.mean(values > 0)),
            }
        )
    return {
        "n_pairs": len(rows),
        "pair_indices": [int(row["source_pair_index"]) for row in rows],
        "mean_pair_coverage_gain_ci": asdict(
            percentile_bootstrap(gains, resamples=resamples, seed=seed)
        ),
        "median_pair_coverage_gain": float(np.median(gains)),
        "fraction_positive_pairs": float(np.mean(gains > 0)),
        "pair_gain_sign_test": exact_two_sided_sign_test(gains),
        "mean_pair_distance_improvement_ci": asdict(
            percentile_bootstrap(improvements, resamples=resamples, seed=seed + 1)
        ),
        "median_headroom_recovery": (
            float(np.median(headrooms)) if headrooms.size else None
        ),
        "median_memory_effect_to_noise_rms": (
            float(np.median(normalized)) if normalized.size else None
        ),
        "tolerance_sensitivity": curve,
        "pair_level": rows,
    }


def main() -> None:
    args = parse_args()
    reports = []
    for path in args.inputs:
        report = json.loads(path.read_text())
        if report.get("experiment") in {
            "E13B_oracle_history_transplant",
            "E13B2_oracle_history_transplant",
        }:
            reports.append(report)
    if not reports:
        raise ValueError("no E13-B/B2 reports were provided")

    audit = _audit_lookup(args.audit)
    tiers = (
        load_pair_quality_tiers(args.quality_config)
        if args.quality_config is not None
        else {}
    )
    pair_rows = [_pair_row(report) for report in reports]
    quality_by_source = {
        int(report.get("source_pair_index", report["pair_index"])): _pair_quality(
            report, audit=audit, tiers=tiers
        )
        for report in reports
    }

    subsets: dict[str, list[dict[str, Any]]] = {"all": pair_rows}
    for tier in tiers:
        subsets[tier] = [
            row
            for row in pair_rows
            if quality_by_source.get(row["source_pair_index"], {}).get(tier, False)
        ]

    output = {
        "experiment": "E13B2_oracle_history_transplant_pair_aggregate",
        "n_input_reports": len(reports),
        "inference_unit": (
            "the two transplant directions are averaged within pair; bootstrap "
            "and sign tests use independent pairs"
        ),
        "support_definition": (
            "native-history-conditioned action support; not ground-truth task correctness"
        ),
        "subsets": {
            name: _summarize(
                rows,
                resamples=args.bootstrap_resamples,
                seed=args.seed + 1_000 * offset,
            )
            for offset, (name, rows) in enumerate(subsets.items())
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
