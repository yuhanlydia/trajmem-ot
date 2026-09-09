#!/usr/bin/env python3
"""E13-D: recover native-history support by masking an incorrect donor history."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from trajmem_ot.hypothesis_metrics import (
    calibrated_set_coverage,
    calibrated_tolerance,
    headroom_recovery,
)
from trajmem_ot.memory_views import (
    allocation_grid,
    contiguous_history_mask_views,
    matched_variance_decomposition,
)
from trajmem_ot.robomme_jax import extract_observation_history, replace_observation_history
from trajmem_ot.robomme_runtime import load_runtime_states


def _parse_int_list(value: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--pair-index", type=int, required=True)
    parser.add_argument("--preset", choices=("16gb", "24gb"), default="16gb")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--config", default="mme_vla_suite")
    parser.add_argument("--history-config")
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--view-counts", type=_parse_int_list)
    parser.add_argument("--keep-fractions", default="0.25,0.5,0.75")
    parser.add_argument("--primary-view-count", type=int, default=4)
    parser.add_argument("--primary-keep-fraction", type=float, default=0.5)
    parser.add_argument("--noise-blocks", type=int, default=3)
    parser.add_argument("--reference-samples", type=int, default=4)
    parser.add_argument("--target-samples", type=int, default=4)
    parser.add_argument("--tolerance-quantile", type=float, default=0.95)
    parser.add_argument("--tolerance-multiplier", type=float, default=1.25)
    parser.add_argument("--robot-action-dim", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path)
    return parser.parse_args()


def _parse_fractions(value: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not values or any(not 0.0 < item <= 1.0 for item in values):
        raise ValueError("keep-fractions must lie in (0,1]")
    return values


def _sample(
    policy: Any,
    observation: Any,
    *,
    model: Any,
    seed: int,
    num_steps: int,
    robot_action_dim: int,
) -> np.ndarray:
    import jax
    import jax.numpy as jnp

    noise_key, model_key = jax.random.split(jax.random.key(seed))
    noise = jax.random.normal(
        noise_key,
        (1, int(model.action_horizon), int(model.action_dim)),
        dtype=jnp.float32,
    )
    actions = policy._sample_actions(model_key, observation, num_steps=num_steps, noise=noise)
    actions.block_until_ready()
    return np.asarray(actions[0], dtype=np.float32)[..., :robot_action_dim]


def _sample_set(policy, observation, *, model, seeds, args):
    return np.stack(
        [
            _sample(
                policy,
                observation,
                model=model,
                seed=seed,
                num_steps=args.num_steps,
                robot_action_dim=args.robot_action_dim,
            )
            for seed in seeds
        ]
    )


def _run_direction(label: str, context_state: Any, donor_state: Any, args) -> tuple[dict, dict]:
    import jax.numpy as jnp

    policy = context_state.policy
    model = policy._model
    correct_observation = context_state.observation
    donor_bundle = extract_observation_history(donor_state.observation, representation="static")
    wrong_observation = replace_observation_history(correct_observation, donor_bundle)
    wrong_mask = np.asarray(wrong_observation.static_mask[0], dtype=bool)

    seed_offset = 10_000 if label.startswith("context_a") else 20_000
    reference = _sample_set(
        policy,
        correct_observation,
        model=model,
        seeds=[args.seed + seed_offset + 100 + i for i in range(args.reference_samples)],
        args=args,
    )
    targets = _sample_set(
        policy,
        correct_observation,
        model=model,
        seeds=[args.seed + seed_offset + 200 + i for i in range(args.target_samples)],
        args=args,
    )
    tolerance = calibrated_tolerance(
        reference,
        targets,
        tolerance_quantile=args.tolerance_quantile,
        tolerance_multiplier=args.tolerance_multiplier,
    )

    total_budget = 8 if args.preset == "16gb" else 32
    default_views = (1, 2, 4) if total_budget == 8 else (1, 4, 8)
    allocations = allocation_grid(
        total_budget, memory_view_counts=args.view_counts or default_views
    )
    if allocations[0][0] != 1 or sum(views == 1 for views, _ in allocations) != 1:
        raise ValueError("view-counts must start with exactly one one-view baseline")
    fractions = _parse_fractions(args.keep_fractions)
    arrays: dict[str, np.ndarray] = {
        f"{label}_reference": reference,
        f"{label}_targets": targets,
    }
    allocation_rows = []
    baseline_by_block: list[float] | None = None

    for views, noises in allocations:
        fraction_rows = []
        used_fractions = (1.0,) if views == 1 else fractions
        for keep_fraction in used_fractions:
            if views == 1:
                masks = wrong_mask[None, ...]
            else:
                masks = contiguous_history_mask_views(
                    wrong_mask,
                    view_count=views,
                    keep_fraction=keep_fraction,
                    include_full=True,
                )
            observations = [
                replace_observation_history(
                    wrong_observation,
                    {"static_mask": jnp.asarray(mask, dtype=bool)[None, ...]},
                )
                for mask in masks
            ]
            block_rows = []
            for block_index in range(args.noise_blocks):
                actions = np.stack(
                    [
                        np.stack(
                            [
                                _sample(
                                    policy,
                                    observation,
                                    model=model,
                                    seed=args.seed
                                    + seed_offset
                                    + 30_000
                                    + 1_000 * block_index
                                    + noise_index,
                                    num_steps=args.num_steps,
                                    robot_action_dim=args.robot_action_dim,
                                )
                                for noise_index in range(noises)
                            ]
                        )
                        for observation in observations
                    ]
                )
                candidates = actions.reshape(
                    actions.shape[0] * actions.shape[1], *actions.shape[2:]
                )
                coverage = calibrated_set_coverage(
                    candidates, targets, tolerance=tolerance
                )
                decomposition = matched_variance_decomposition(actions)
                block_rows.append(
                    {
                        "block_index": block_index,
                        "coverage": asdict(coverage),
                        "memory_main_fraction": decomposition.memory_fraction,
                        "noise_main_fraction": decomposition.noise_fraction,
                        "interaction_fraction": decomposition.interaction_fraction,
                    }
                )
                arrays[
                    f"{label}_B{views}_N{noises}_k{keep_fraction:g}_block{block_index}"
                ] = actions
            coverages = [row["coverage"]["coverage_fraction"] for row in block_rows]
            if views == 1:
                baseline_by_block = coverages
            if baseline_by_block is None:
                raise RuntimeError("allocation grid must include the one-view baseline first")
            gains = [
                float(value - baseline_by_block[index])
                for index, value in enumerate(coverages)
            ]
            for index, gain in enumerate(gains):
                block_rows[index]["coverage_gain_vs_wrong_noise_only"] = gain
            fraction_rows.append(
                {
                    "keep_fraction": keep_fraction,
                    "valid_tokens_per_view": [int(mask.sum()) for mask in masks],
                    "mean_coverage": float(np.mean(coverages)),
                    "mean_coverage_gain_vs_wrong_noise_only": float(np.mean(gains)),
                    "mean_headroom_recovery": float(
                        np.mean(
                            [
                                value
                                for value in (
                                    headroom_recovery(
                                        noise_coverage=baseline_by_block[i],
                                        branched_coverage=coverages[i],
                                    )
                                    for i in range(len(coverages))
                                )
                                if value is not None
                            ]
                        )
                    )
                    if any(value < 1.0 for value in baseline_by_block)
                    else None,
                    "blocks": block_rows,
                }
            )
        allocation_rows.append(
            {
                "memory_views": views,
                "noises_per_view": noises,
                "fractions": fraction_rows,
            }
        )

    try:
        primary_allocation = next(
            allocation
            for allocation in allocation_rows
            if allocation["memory_views"] == args.primary_view_count
        )
        primary_row = next(
            row
            for row in primary_allocation["fractions"]
            if np.isclose(row["keep_fraction"], args.primary_keep_fraction)
        )
    except StopIteration as exc:
        raise ValueError(
            "the preregistered primary view count/keep fraction is absent from "
            "--view-counts/--keep-fractions"
        ) from exc
    return (
        {
            "direction": label,
            "context_index": context_state.index,
            "donor_history_index": donor_state.index,
            "native_support_tolerance": tolerance,
            "allocations": allocation_rows,
            "primary_view_count": args.primary_view_count,
            "primary_keep_fraction": args.primary_keep_fraction,
            "primary_readout_coverage_gain": primary_row[
                "mean_coverage_gain_vs_wrong_noise_only"
            ],
            "sensitivity_only": (
                "non-primary view counts and keep fractions must not be selected "
                "per test pair using target coverage"
            ),
        },
        arrays,
    )


def main() -> None:
    args = parse_args()
    if args.noise_blocks <= 0:
        raise ValueError("noise-blocks must be positive")
    pairs = json.loads(args.pairs.read_text())
    if args.pair_index < 0 or args.pair_index >= len(pairs):
        raise IndexError("pair-index is outside the pair file")
    pair = pairs[args.pair_index]
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault(
        "XLA_PYTHON_CLIENT_MEM_FRACTION", "0.75" if args.preset == "16gb" else "0.88"
    )
    state_a, state_b = load_runtime_states(
        checkpoint=args.checkpoint,
        data=args.data,
        indices=(int(pair["a"]), int(pair["b"])),
        seed=args.seed,
        train_config_name=args.config,
        history_config_name=args.history_config,
    )
    a_report, a_arrays = _run_direction(
        "context_a_correct_a_wrong_b", state_a, state_b, args
    )
    b_report, b_arrays = _run_direction(
        "context_b_correct_b_wrong_a", state_b, state_a, args
    )
    report = {
        "experiment": "E13D_pair_readout_recovery",
        "pair_index": args.pair_index,
        "source_pair_index": int(pair.get("source_pair_index", args.pair_index)),
        "preset": args.preset,
        "noise_blocks": args.noise_blocks,
        "directions": [a_report, b_report],
        "pair_mean_primary_readout_coverage_gain": float(
            np.mean(
                [
                    a_report["primary_readout_coverage_gain"],
                    b_report["primary_readout_coverage_gain"],
                ]
            )
        ),
        "claim_scope": (
            "parameter-free contiguous readout masks applied to an incorrect donor "
            "history; native-history support recovery, not task success"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    artifact_path = args.artifacts or args.output.with_suffix(".npz")
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(artifact_path, **a_arrays, **b_arrays)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
