#!/usr/bin/env python3
"""E13-C2: matched readout branching with replicated noise blocks and support."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from trajmem_ot.hypothesis_metrics import calibrated_set_coverage, calibrated_tolerance
from trajmem_ot.memory_views import (
    allocation_grid,
    contiguous_history_mask_views,
    matched_variance_decomposition,
    nearest_target_coverage,
    variance_decomposition,
)
from trajmem_ot.robomme_jax import replace_observation_history
from trajmem_ot.robomme_runtime import load_runtime_state


def _parse_int_list(value: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--preset", choices=("16gb", "24gb"), default="16gb")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--config", default="mme_vla_suite")
    parser.add_argument("--history-config")
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--keep-fraction", type=float, default=0.5)
    parser.add_argument(
        "--exclude-full-view",
        action="store_true",
        help="use only partial windows; default retains the full-memory view",
    )
    parser.add_argument("--view-counts", type=_parse_int_list)
    parser.add_argument("--noise-blocks", type=int, default=1)
    parser.add_argument("--robot-action-dim", type=int, default=8)
    parser.add_argument("--target-actions", type=Path)
    parser.add_argument("--coverage-tolerance", type=float, default=1.0)
    parser.add_argument("--calibrate-native-support", action="store_true")
    parser.add_argument("--reference-samples", type=int, default=4)
    parser.add_argument("--target-samples", type=int, default=4)
    parser.add_argument("--tolerance-quantile", type=float, default=0.95)
    parser.add_argument("--tolerance-multiplier", type=float, default=1.25)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path)
    return parser.parse_args()


def _sample(
    policy: Any, observation: Any, *, model: Any, seed: int, num_steps: int
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
    return np.asarray(actions[0], dtype=np.float32)


def _sample_set(
    policy: Any,
    observation: Any,
    *,
    model: Any,
    seeds: list[int],
    num_steps: int,
    robot_action_dim: int,
) -> np.ndarray:
    return np.stack(
        [
            _sample(policy, observation, model=model, seed=seed, num_steps=num_steps)[
                ..., :robot_action_dim
            ]
            for seed in seeds
        ],
        axis=0,
    )


def _decomposition(actions: np.ndarray) -> dict[str, float]:
    legacy = variance_decomposition(actions)
    matched = matched_variance_decomposition(actions)
    return {
        "total_variance": legacy.total,
        "within_noise_variance": legacy.within_noise,
        "between_memory_variance": legacy.between_memory,
        "memory_variance_fraction": legacy.memory_fraction,
        "matched_memory_main_variance": matched.memory_main,
        "matched_noise_main_variance": matched.noise_main,
        "matched_interaction_variance": matched.interaction,
        "matched_memory_main_fraction": matched.memory_fraction,
        "matched_noise_main_fraction": matched.noise_fraction,
        "matched_interaction_fraction": matched.interaction_fraction,
    }


def _mean_metrics(blocks: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: float(np.mean([block[key] for block in blocks]))
        for key in blocks[0]
    }


def main() -> None:
    args = parse_args()
    if args.noise_blocks <= 0:
        raise ValueError("noise-blocks must be positive")
    if args.reference_samples <= 0 or args.target_samples <= 0:
        raise ValueError("reference-samples and target-samples must be positive")

    total_budget = 8 if args.preset == "16gb" else 32
    default_views = (1, 2, 4) if total_budget == 8 else (1, 4, 8)
    allocations = allocation_grid(
        total_budget, memory_view_counts=args.view_counts or default_views
    )
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault(
        "XLA_PYTHON_CLIENT_MEM_FRACTION", "0.75" if args.preset == "16gb" else "0.88"
    )

    import jax.numpy as jnp

    runtime = load_runtime_state(
        checkpoint=args.checkpoint,
        data=args.data,
        index=args.index,
        seed=args.seed,
        train_config_name=args.config,
        history_config_name=args.history_config,
    )
    policy = runtime.policy
    model = policy._model
    base_mask = np.asarray(runtime.observation.static_mask[0], dtype=bool)
    external_target = (
        None if args.target_actions is None else np.asarray(np.load(args.target_actions))
    )

    native_reference = native_targets = None
    native_tolerance = None
    if args.calibrate_native_support:
        native_reference = _sample_set(
            policy,
            runtime.observation,
            model=model,
            seeds=[args.seed + 40_000 + i for i in range(args.reference_samples)],
            num_steps=args.num_steps,
            robot_action_dim=args.robot_action_dim,
        )
        native_targets = _sample_set(
            policy,
            runtime.observation,
            model=model,
            seeds=[args.seed + 41_000 + i for i in range(args.target_samples)],
            num_steps=args.num_steps,
            robot_action_dim=args.robot_action_dim,
        )
        native_tolerance = calibrated_tolerance(
            native_reference,
            native_targets,
            tolerance_quantile=args.tolerance_quantile,
            tolerance_multiplier=args.tolerance_multiplier,
        )

    summaries: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {"base_mask": base_mask}
    if native_reference is not None and native_targets is not None:
        arrays["native_reference"] = native_reference
        arrays["native_targets"] = native_targets

    for views_count, noises_count in allocations:
        if views_count == 1:
            masks = base_mask[None, ...]
        else:
            masks = contiguous_history_mask_views(
                base_mask,
                view_count=views_count,
                keep_fraction=args.keep_fraction,
                include_full=not args.exclude_full_view,
            )
        observations = [
            replace_observation_history(
                runtime.observation,
                {"static_mask": jnp.asarray(mask, dtype=bool)[None, ...]},
            )
            for mask in masks
        ]

        block_rows: list[dict[str, Any]] = []
        for block_index in range(args.noise_blocks):
            action_rows = []
            for noise_index in range(noises_count):
                seed = args.seed + 30_000 + 1_000 * block_index + noise_index
                action_rows.append(
                    np.stack(
                        [
                            _sample(
                                policy,
                                observation,
                                model=model,
                                seed=seed,
                                num_steps=args.num_steps,
                            )
                            for observation in observations
                        ],
                        axis=0,
                    )
                )
            actions = np.stack(action_rows, axis=0).swapaxes(0, 1)
            robot_actions = actions[..., : args.robot_action_dim]
            block: dict[str, Any] = {
                "block_index": block_index,
                "all_action_channels": _decomposition(actions),
                "robot_action_8d": _decomposition(robot_actions),
            }
            if external_target is not None:
                target_robot = external_target[..., : args.robot_action_dim]
                block["target_coverage"] = asdict(
                    nearest_target_coverage(
                        robot_actions,
                        target_robot,
                        tolerance=args.coverage_tolerance,
                    )
                )
            if native_targets is not None and native_tolerance is not None:
                candidates = robot_actions.reshape(
                    robot_actions.shape[0] * robot_actions.shape[1],
                    *robot_actions.shape[2:],
                )
                block["native_support"] = asdict(
                    calibrated_set_coverage(
                        candidates, native_targets, tolerance=native_tolerance
                    )
                )
            block_rows.append(block)
            key = f"B{views_count}_N{noises_count}_block{block_index}"
            arrays[f"actions_{key}"] = actions

        row: dict[str, Any] = {
            "memory_views": views_count,
            "noises_per_view": noises_count,
            "noise_blocks": args.noise_blocks,
            "total_trajectories_per_block": views_count * noises_count,
            "total_policy_calls": views_count * noises_count * args.noise_blocks,
            "keep_fraction": args.keep_fraction if views_count > 1 else 1.0,
            "includes_full_memory_view": bool(
                views_count == 1 or not args.exclude_full_view
            ),
            "valid_tokens_per_view": [int(mask.sum()) for mask in masks],
            "all_action_channels": _mean_metrics(
                [block["all_action_channels"] for block in block_rows]
            ),
            "robot_action_8d": _mean_metrics(
                [block["robot_action_8d"] for block in block_rows]
            ),
            "blocks": block_rows,
        }
        summaries.append(row)
        arrays[f"masks_B{views_count}_N{noises_count}"] = masks

    if native_targets is not None:
        baseline = next(row for row in summaries if row["memory_views"] == 1)
        for row in summaries:
            gains = []
            for block_index, block in enumerate(row["blocks"]):
                baseline_coverage = float(
                    baseline["blocks"][block_index]["native_support"]["coverage_fraction"]
                )
                gain = float(block["native_support"]["coverage_fraction"]) - baseline_coverage
                block["native_support"]["coverage_gain_vs_noise_only"] = gain
                gains.append(gain)
            row["native_support"] = {
                "mean_coverage_fraction": float(
                    np.mean(
                        [
                            block["native_support"]["coverage_fraction"]
                            for block in row["blocks"]
                        ]
                    )
                ),
                "mean_coverage_gain_vs_noise_only": float(np.mean(gains)),
                "coverage_gains_by_block": gains,
            }

    report = {
        "experiment": "E13C2_history_readout_mask_branching",
        "checkpoint": str(args.checkpoint),
        "data": str(args.data),
        "index": args.index,
        "preset": args.preset,
        "robot_action_dim": args.robot_action_dim,
        "matched_noise_seeds": True,
        "noise_blocks": args.noise_blocks,
        "allocations": summaries,
        "native_support_calibration": (
            None
            if native_tolerance is None
            else {
                "tolerance": native_tolerance,
                "reference_samples": args.reference_samples,
                "target_samples": args.target_samples,
                "tolerance_quantile": args.tolerance_quantile,
                "tolerance_multiplier": args.tolerance_multiplier,
            }
        ),
        "claim_scope": (
            "deployment-faithful readout branching with matched two-way variance "
            "and optional native-history support coverage; not task success or "
            "calibrated belief uncertainty"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    artifact_path = args.artifacts or args.output.with_suffix(".npz")
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(artifact_path, **arrays)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
