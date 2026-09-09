#!/usr/bin/env python3
"""E13-C: history-readout mask branching versus diffusion-noise branching."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from trajmem_ot.memory_views import (
    allocation_grid,
    contiguous_history_mask_views,
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
        help="use only partial windows; by default every B>1 allocation retains the full-memory view",
    )
    parser.add_argument("--view-counts", type=_parse_int_list)
    parser.add_argument("--robot-action-dim", type=int, default=8)
    parser.add_argument("--target-actions", type=Path)
    parser.add_argument("--coverage-tolerance", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path)
    return parser.parse_args()


def _sample(policy: Any, observation: Any, *, model: Any, seed: int, num_steps: int) -> np.ndarray:
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


def _decomposition_dict(value) -> dict[str, float]:
    return {
        "total_variance": value.total,
        "within_noise_variance": value.within_noise,
        "between_memory_variance": value.between_memory,
        "memory_variance_fraction": value.memory_fraction,
    }


def main() -> None:
    args = parse_args()
    total_budget = 8 if args.preset == "16gb" else 32
    default_views = (1, 2, 4) if total_budget == 8 else (1, 4, 8)
    view_counts = args.view_counts or default_views
    allocations = allocation_grid(total_budget, memory_view_counts=view_counts)
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
    target = None if args.target_actions is None else np.asarray(np.load(args.target_actions))

    summaries = []
    arrays: dict[str, np.ndarray] = {"base_mask": base_mask}
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
        action_rows = []
        for noise_index in range(noises_count):
            seed = args.seed + 30_000 + noise_index
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
        all_dec = variance_decomposition(actions)
        robot_dec = variance_decomposition(robot_actions)
        row: dict[str, Any] = {
            "memory_views": views_count,
            "noises_per_view": noises_count,
            "total_trajectories": views_count * noises_count,
            "keep_fraction": args.keep_fraction if views_count > 1 else 1.0,
            "includes_full_memory_view": bool(views_count == 1 or not args.exclude_full_view),
            "valid_tokens_per_view": [int(mask.sum()) for mask in masks],
            "all_action_channels": _decomposition_dict(all_dec),
            "robot_action_8d": _decomposition_dict(robot_dec),
        }
        if target is not None:
            target_robot = target[..., : args.robot_action_dim]
            coverage = nearest_target_coverage(
                robot_actions, target_robot, tolerance=args.coverage_tolerance
            )
            row["target_coverage"] = {
                "covered": coverage.covered,
                "covered_views": coverage.covered_views,
                "covered_samples": coverage.covered_samples,
                "total_samples": coverage.total_samples,
                "min_distance": coverage.min_distance,
            }
        summaries.append(row)
        key = f"B{views_count}_N{noises_count}"
        arrays[f"masks_{key}"] = masks
        arrays[f"actions_{key}"] = actions

    report = {
        "experiment": "E13C_history_readout_mask_branching",
        "checkpoint": str(args.checkpoint),
        "data": str(args.data),
        "index": args.index,
        "preset": args.preset,
        "robot_action_dim": args.robot_action_dim,
        "allocations": summaries,
        "claim_scope": (
            "deployment-faithful readout branching with unchanged memory values; "
            "not learned FP32 attention bias and not environment success"
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
