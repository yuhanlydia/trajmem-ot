#!/usr/bin/env python3
"""E13-B: oracle full-history transplant versus noise-only sampling.

This is a phenomenon/upper-bound experiment. For a matched pair of histories,
the current observation, current robot state, instruction, policy weights, and
noise distribution are fixed. Only the complete history bundle is swapped.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from trajmem_ot.hypothesis_metrics import compare_hypothesis_coverage
from trajmem_ot.robomme_jax import (
    extract_observation_history,
    replace_observation_history,
)
from trajmem_ot.robomme_runtime import load_runtime_states


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
    parser.add_argument("--robot-action-dim", type=int, default=8)
    parser.add_argument("--reference-samples", type=int, default=4)
    parser.add_argument("--target-samples", type=int, default=4)
    parser.add_argument("--tolerance-quantile", type=float, default=0.95)
    parser.add_argument("--tolerance-multiplier", type=float, default=1.25)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path)
    return parser.parse_args()


def _to_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    array = np.asarray(value)
    return str(array.item()) if array.shape == () else str(value)


def _normalized_image_mae(left: Any, right: Any) -> float:
    a = np.asarray(left, dtype=np.float32)
    b = np.asarray(right, dtype=np.float32)
    if a.shape != b.shape:
        return float("inf")
    scale = 255.0 if max(float(np.max(np.abs(a))), float(np.max(np.abs(b)))) > 1.5 else 1.0
    return float(np.mean(np.abs(a - b)) / scale)


def _sample_set(
    policy: Any,
    observation: Any,
    *,
    model: Any,
    seeds: list[int],
    num_steps: int,
    robot_action_dim: int,
) -> np.ndarray:
    import jax
    import jax.numpy as jnp

    rows = []
    for seed in seeds:
        noise_key, model_key = jax.random.split(jax.random.key(seed))
        noise = jax.random.normal(
            noise_key,
            (1, int(model.action_horizon), int(model.action_dim)),
            dtype=jnp.float32,
        )
        actions = policy._sample_actions(
            model_key,
            observation,
            num_steps=num_steps,
            noise=noise,
        )
        actions.block_until_ready()
        rows.append(np.asarray(actions[0], dtype=np.float32)[..., :robot_action_dim])
    return np.stack(rows, axis=0)


def _variance(actions: np.ndarray) -> float:
    flat = np.asarray(actions, dtype=np.float64).reshape(actions.shape[0], -1)
    mean = flat.mean(axis=0, keepdims=True)
    return float(np.mean(np.sum(np.square(flat - mean), axis=-1)))


def _run_direction(
    *,
    label: str,
    context_state: Any,
    donor_state: Any,
    budget: int,
    args: argparse.Namespace,
    seed_offset: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    policy = context_state.policy
    model = policy._model
    correct_observation = context_state.observation
    donor_bundle = extract_observation_history(donor_state.observation, representation="static")
    wrong_observation = replace_observation_history(correct_observation, donor_bundle)

    reference_seeds = [args.seed + seed_offset + 100 + i for i in range(args.reference_samples)]
    target_seeds = [args.seed + seed_offset + 200 + i for i in range(args.target_samples)]
    candidate_seeds = [args.seed + seed_offset + 300 + i for i in range(budget)]
    # Reuse noise seeds across hypotheses so the matched-compute comparison
    # isolates the history intervention rather than an unrelated noise draw.
    branch_seeds = candidate_seeds[: budget // 2]
    paired_seeds = branch_seeds[: min(4, len(branch_seeds))]

    correct_reference = _sample_set(
        policy,
        correct_observation,
        model=model,
        seeds=reference_seeds,
        num_steps=args.num_steps,
        robot_action_dim=args.robot_action_dim,
    )
    correct_targets = _sample_set(
        policy,
        correct_observation,
        model=model,
        seeds=target_seeds,
        num_steps=args.num_steps,
        robot_action_dim=args.robot_action_dim,
    )
    noise_only_wrong = _sample_set(
        policy,
        wrong_observation,
        model=model,
        seeds=candidate_seeds,
        num_steps=args.num_steps,
        robot_action_dim=args.robot_action_dim,
    )
    correct_only = _sample_set(
        policy,
        correct_observation,
        model=model,
        seeds=candidate_seeds,
        num_steps=args.num_steps,
        robot_action_dim=args.robot_action_dim,
    )
    # Reuse already sampled, seed-aligned trajectories. Besides reducing GPU
    # work, this makes the only difference between paired branches the history.
    branched_wrong = noise_only_wrong[: len(branch_seeds)]
    branched_correct = correct_only[: len(branch_seeds)]
    branched = np.concatenate([branched_wrong, branched_correct], axis=0)

    paired_count = len(paired_seeds)
    paired_correct = correct_only[:paired_count]
    paired_wrong = noise_only_wrong[:paired_count]
    paired_effects = np.linalg.norm(
        (paired_correct - paired_wrong).reshape(paired_correct.shape[0], -1), axis=1
    )

    comparison = compare_hypothesis_coverage(
        correct_reference,
        correct_targets,
        noise_only_wrong,
        branched,
        correct_only=correct_only,
        tolerance_quantile=args.tolerance_quantile,
        tolerance_multiplier=args.tolerance_multiplier,
    )
    report = {
        "direction": label,
        "context_index": context_state.index,
        "donor_history_index": donor_state.index,
        "budget_per_condition": budget,
        "noise_only_allocation": {"memory_views": 1, "noises_per_view": budget},
        "oracle_branch_allocation": {"memory_views": 2, "noises_per_view": budget // 2},
        "coverage": asdict(comparison),
        "paired_memory_effect_mean": float(np.mean(paired_effects)),
        "paired_memory_effect_median": float(np.median(paired_effects)),
        "wrong_noise_variance": _variance(noise_only_wrong),
        "correct_noise_variance": _variance(correct_only),
    }
    arrays = {
        f"{label}_correct_reference": correct_reference,
        f"{label}_correct_targets": correct_targets,
        f"{label}_noise_only_wrong": noise_only_wrong,
        f"{label}_correct_only": correct_only,
        f"{label}_branched_wrong": branched_wrong,
        f"{label}_branched_correct": branched_correct,
        f"{label}_paired_correct": paired_correct,
        f"{label}_paired_wrong": paired_wrong,
    }
    return report, arrays


def main() -> None:
    args = parse_args()
    budget = 8 if args.preset == "16gb" else 32
    if budget % 2:
        raise ValueError("total budget must be divisible by two")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault(
        "XLA_PYTHON_CLIENT_MEM_FRACTION", "0.75" if args.preset == "16gb" else "0.88"
    )

    pairs = json.loads(args.pairs.read_text())
    if args.pair_index < 0 or args.pair_index >= len(pairs):
        raise IndexError(f"pair-index {args.pair_index} outside [0, {len(pairs) - 1}]")
    pair = pairs[args.pair_index]
    index_a, index_b = int(pair["a"]), int(pair["b"])
    state_a, state_b = load_runtime_states(
        checkpoint=args.checkpoint,
        data=args.data,
        indices=(index_a, index_b),
        seed=args.seed,
        train_config_name=args.config,
        history_config_name=args.history_config,
    )

    prompt_a = _to_string(state_a.item.get("prompt"))
    prompt_b = _to_string(state_b.item.get("prompt"))
    context = {
        "prompt_match": prompt_a == prompt_b,
        "prompt_a": prompt_a,
        "prompt_b": prompt_b,
        "front_image_mae": _normalized_image_mae(
            state_a.item.get("image"), state_b.item.get("image")
        ),
        "wrist_image_mae": _normalized_image_mae(
            state_a.item.get("wrist_image"), state_b.item.get("wrist_image")
        ),
        "state_l2": float(
            np.linalg.norm(
                np.asarray(state_a.item.get("state"), dtype=np.float64)
                - np.asarray(state_b.item.get("state"), dtype=np.float64)
            )
        ),
        "pair_metadata": pair,
    }

    a_report, a_arrays = _run_direction(
        label="context_a_correct_a_wrong_b",
        context_state=state_a,
        donor_state=state_b,
        budget=budget,
        args=args,
        seed_offset=10_000,
    )
    b_report, b_arrays = _run_direction(
        label="context_b_correct_b_wrong_a",
        context_state=state_b,
        donor_state=state_a,
        budget=budget,
        args=args,
        seed_offset=20_000,
    )
    gains = [
        a_report["coverage"]["coverage_gain"],
        b_report["coverage"]["coverage_gain"],
    ]
    report = {
        "experiment": "E13B_oracle_history_transplant",
        "pair_index": args.pair_index,
        "preset": args.preset,
        "robot_action_dim": args.robot_action_dim,
        "num_steps": args.num_steps,
        "context_diagnostics": context,
        "directions": [a_report, b_report],
        "mean_oracle_coverage_gain": float(np.mean(gains)),
        "claim_scope": (
            "oracle upper-bound test of shared-memory conditional-support failure; "
            "not a deployable hypothesis generator and not environment success"
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
