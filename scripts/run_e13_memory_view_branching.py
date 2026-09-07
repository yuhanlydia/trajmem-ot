#!/usr/bin/env python3
"""E13: matched-compute memory-view branching versus diffusion-noise branching."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from trajmem_ot.memory_basis import normalize_basis, random_rank_one_basis
from trajmem_ot.memory_views import (
    allocation_grid,
    build_memory_views,
    hypothesis_coefficients,
    nearest_target_coverage,
    variance_decomposition,
)
from trajmem_ot.presets import get_compute_preset
from trajmem_ot.robomme_jax import build_fixed_noise_action_problem
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
    parser.add_argument("--memory-field", choices=("static_image_emb", "recur_image_emb"), default="static_image_emb")
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--basis", type=Path, help="NPZ containing key 'basis'; random controls are used when omitted")
    parser.add_argument("--relative-radius", type=float, default=2.5e-4)
    parser.add_argument("--total-budget", type=int)
    parser.add_argument("--view-counts", type=_parse_int_list)
    parser.add_argument("--target-actions", type=Path, help="optional normalized [H,A] target for coverage")
    parser.add_argument("--coverage-tolerance", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preset = get_compute_preset(args.preset)
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", str(preset.xla_memory_fraction))

    import jax
    import jax.numpy as jnp

    runtime = load_runtime_state(
        checkpoint=args.checkpoint,
        data=args.data,
        index=args.index,
        seed=args.seed,
        train_config_name=args.config,
        history_config_name=args.history_config,
    )
    model = runtime.policy._model
    # Build one temporary problem only to expose the unbatched base memory.
    initial_noise = jax.random.normal(
        jax.random.key(args.seed + 501),
        (1, int(model.action_horizon), int(model.action_dim)),
        dtype=jnp.float32,
    )
    base_problem = build_fixed_noise_action_problem(
        runtime.policy,
        runtime.observation,
        noise=initial_noise,
        rng=jax.random.key(args.seed + 502),
        memory_field=args.memory_field,
        num_steps=args.num_steps,
    )
    memory = np.asarray(base_problem.memory, dtype=np.float32)
    if args.basis is None:
        basis = random_rank_one_basis(
            memory.shape, count=preset.basis_directions, seed=args.seed + 503
        )
        basis_source = "random_control"
    else:
        basis = normalize_basis(np.asarray(np.load(args.basis)["basis"]))
        if basis.shape[1:] != memory.shape:
            raise ValueError(f"basis shape {basis.shape[1:]} != memory shape {memory.shape}")
        basis = basis[: min(preset.basis_directions, basis.shape[0])]
        basis_source = str(args.basis)

    total_budget = args.total_budget or preset.total_trajectories
    if args.view_counts is None:
        view_counts = (1, 2, 4, 8) if total_budget == 8 else (1, 4, 8, 32)
    else:
        view_counts = args.view_counts
    allocations = allocation_grid(total_budget, memory_view_counts=view_counts)
    target = None if args.target_actions is None else np.asarray(np.load(args.target_actions))

    summaries = []
    arrays = {"memory": memory, "basis": basis}
    for views_count, noises_count in allocations:
        if views_count == 1:
            coefficients = np.zeros((1, basis.shape[0]), dtype=np.float64)
        else:
            coefficients = hypothesis_coefficients(
                view_count=views_count,
                basis_size=basis.shape[0],
                seed=args.seed + views_count,
                include_base=bool(views_count % 2 == 1),
            )
        view_batch = build_memory_views(
            memory,
            basis,
            coefficients,
            relative_radius=args.relative_radius,
        )
        action_rows = []
        for noise_index in range(noises_count):
            noise_key, model_key = jax.random.split(
                jax.random.key(args.seed + 10_000 + noise_index)
            )
            noise = jax.random.normal(
                noise_key,
                (1, int(model.action_horizon), int(model.action_dim)),
                dtype=jnp.float32,
            )
            problem = build_fixed_noise_action_problem(
                runtime.policy,
                runtime.observation,
                noise=noise,
                rng=model_key,
                memory_field=args.memory_field,
                num_steps=args.num_steps,
            )
            action_rows.append(
                np.stack(
                    [np.asarray(problem.action_fn(jnp.asarray(view))) for view in view_batch.views],
                    axis=0,
                )
            )
        # loop result is [noise, view, H, A]; experiments use [view, noise, H, A]
        actions = np.stack(action_rows, axis=0).swapaxes(0, 1)
        decomposition = variance_decomposition(actions)
        row = {
            "memory_views": views_count,
            "noises_per_view": noises_count,
            "total_trajectories": int(views_count * noises_count),
            "total_variance": decomposition.total,
            "within_noise_variance": decomposition.within_noise,
            "between_memory_variance": decomposition.between_memory,
            "memory_variance_fraction": decomposition.memory_fraction,
            "max_applied_relative_edit": float(
                view_batch.applied_delta_norms.max() / max(np.linalg.norm(memory), 1e-12)
            ),
        }
        if target is not None:
            coverage = nearest_target_coverage(
                actions, target, tolerance=args.coverage_tolerance
            )
            row["target_coverage"] = {
                "covered": coverage.covered,
                "covered_views": coverage.covered_views,
                "covered_samples": coverage.covered_samples,
                "min_distance": coverage.min_distance,
            }
        summaries.append(row)
        key = f"B{views_count}_N{noises_count}"
        arrays[f"actions_{key}"] = actions
        arrays[f"coefficients_{key}"] = coefficients
        arrays[f"deltas_{key}"] = view_batch.deltas

    report = {
        "experiment": "E13_memory_view_vs_noise_branching",
        "checkpoint": str(args.checkpoint),
        "data": str(args.data),
        "index": args.index,
        "preset": preset.name,
        "basis_source": basis_source,
        "basis_directions": int(basis.shape[0]),
        "relative_radius": args.relative_radius,
        "allocations": summaries,
        "claim_scope": "action-distribution uncertainty decomposition; environment success requires branch rollouts",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    artifact_path = args.artifacts or args.output.with_suffix(".npz")
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(artifact_path, **arrays)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
