#!/usr/bin/env python3
"""E12-S: reward-free recovery using deployed BF16 finite responses, not AD JVP."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

import numpy as np

from trajmem_ot.finite_response import (
    build_finite_response_model,
    solve_finite_response_pullback,
)
from trajmem_ot.memory_basis import normalize_basis, random_rank_one_basis
from trajmem_ot.robomme_jax import build_fixed_noise_action_problem
from trajmem_ot.robomme_runtime import load_runtime_state


def _parse_float_list(value: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive floats")
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
    parser.add_argument("--drop-fraction", type=float, default=0.20)
    parser.add_argument("--basis", choices=("random", "history", "hybrid"), default="hybrid")
    parser.add_argument("--history-basis", type=Path)
    parser.add_argument(
        "--probe-radii",
        type=_parse_float_list,
        default=_parse_float_list("2.5e-4,1e-3,2.5e-3,5e-3"),
    )
    parser.add_argument("--max-update-radius", type=float, default=2.5e-3)
    parser.add_argument("--damping", type=float, default=1e-4)
    parser.add_argument("--response-rank", type=int)
    parser.add_argument("--fresh-noises", type=int, default=4)
    parser.add_argument("--robot-action-dim", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path)
    return parser.parse_args()


def _degrade(
    memory: np.ndarray, valid_mask: np.ndarray | None, fraction: float, seed: int
) -> tuple[np.ndarray, list[int]]:
    if not 0.0 < fraction < 1.0:
        raise ValueError("drop-fraction must be in (0, 1)")
    valid = np.arange(memory.shape[0]) if valid_mask is None else np.flatnonzero(valid_mask)
    if valid.size == 0:
        raise ValueError("history mask contains no valid rows")
    count = max(1, int(round(valid.size * fraction)))
    rng = np.random.default_rng(seed)
    dropped = np.sort(rng.choice(valid, size=min(count, valid.size), replace=False))
    degraded = memory.copy()
    degraded[dropped] = 0.0
    return degraded, dropped.tolist()


def _load_basis(
    args: argparse.Namespace,
    memory: np.ndarray,
    corruption: np.ndarray,
    count: int,
) -> np.ndarray:
    random = random_rank_one_basis(memory.shape, count=count, seed=args.seed + 211)
    if args.basis == "random":
        return random
    if args.basis == "history":
        if args.history_basis is None:
            raise ValueError("--history-basis is required for --basis=history")
        history = normalize_basis(np.asarray(np.load(args.history_basis)["basis"]))
        if history.shape[1:] != memory.shape:
            raise ValueError("history basis shape does not match memory")
        return history[: min(count, history.shape[0])]
    direction = corruption / max(np.linalg.norm(corruption), 1e-12)
    if np.linalg.norm(direction) <= 1e-12:
        raise ValueError("memory degradation produced no nonzero direction")
    return normalize_basis(np.concatenate([direction[None, ...], random[: count - 1]], axis=0))


def _recovery(clean: np.ndarray, degraded: np.ndarray, candidate: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(degraded - clean)), 1e-12)
    return float(1.0 - np.linalg.norm(candidate - clean) / denominator)


def main() -> None:
    args = parse_args()
    basis_count = 8 if args.preset == "16gb" else 16
    default_rank = 4 if args.preset == "16gb" else 8
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault(
        "XLA_PYTHON_CLIENT_MEM_FRACTION", "0.75" if args.preset == "16gb" else "0.88"
    )

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
    noise_key, model_key = jax.random.split(jax.random.key(args.seed + 301))
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
        memory_field="static_image_emb",
        num_steps=args.num_steps,
    )
    clean_np = np.asarray(problem.memory, dtype=np.float32)
    mask = np.asarray(runtime.observation.static_mask[0], dtype=bool)
    degraded_np, dropped = _degrade(clean_np, mask, args.drop_fraction, args.seed + args.index)
    degraded = jnp.asarray(degraded_np, dtype=problem.memory.dtype)
    clean_actions = np.asarray(problem.action_fn(problem.memory), dtype=np.float32)[
        ..., : args.robot_action_dim
    ]
    degraded_actions = np.asarray(problem.action_fn(degraded), dtype=np.float32)[
        ..., : args.robot_action_dim
    ]
    target_delta = clean_actions - degraded_actions
    basis = _load_basis(args, degraded_np, clean_np - degraded_np, basis_count)

    reports = []
    artifact_arrays: dict[str, np.ndarray] = {
        "clean_memory": clean_np,
        "degraded_memory": degraded_np,
        "basis": basis,
        "clean_actions": clean_actions,
        "degraded_actions": degraded_actions,
    }
    for radius_index, probe_radius in enumerate(args.probe_radii):
        response = build_finite_response_model(
            problem.action_fn,
            degraded,
            basis,
            relative_radius=probe_radius,
            robot_action_dim=args.robot_action_dim,
        )
        rank = min(args.response_rank or default_rank, min(response.response_matrix.shape))
        pullback = solve_finite_response_pullback(
            response,
            degraded,
            target_delta,
            damping=args.damping,
            rank=rank,
            max_relative_norm=args.max_update_radius,
        )
        edited = jnp.asarray(pullback.quantized_memory, dtype=problem.memory.dtype)
        negative = jnp.asarray(degraded_np - pullback.applied_delta, dtype=problem.memory.dtype)
        rng = np.random.default_rng(args.seed + 900 + radius_index)
        random_delta = rng.normal(size=degraded_np.shape).astype(np.float32)
        random_delta *= np.linalg.norm(pullback.applied_delta) / max(
            np.linalg.norm(random_delta), 1e-12
        )
        random_memory = jnp.asarray(degraded_np + random_delta, dtype=problem.memory.dtype)

        edited_actions = np.asarray(problem.action_fn(edited), dtype=np.float32)[
            ..., : args.robot_action_dim
        ]
        negative_actions = np.asarray(problem.action_fn(negative), dtype=np.float32)[
            ..., : args.robot_action_dim
        ]
        random_actions = np.asarray(problem.action_fn(random_memory), dtype=np.float32)[
            ..., : args.robot_action_dim
        ]

        fresh_recoveries = []
        for offset in range(args.fresh_noises):
            fresh_noise_key, fresh_model_key = jax.random.split(
                jax.random.key(args.seed + 2_000 + offset)
            )
            fresh_noise = jax.random.normal(
                fresh_noise_key,
                (1, int(model.action_horizon), int(model.action_dim)),
                dtype=jnp.float32,
            )
            fresh_problem = build_fixed_noise_action_problem(
                runtime.policy,
                runtime.observation,
                noise=fresh_noise,
                rng=fresh_model_key,
                memory_field="static_image_emb",
                num_steps=args.num_steps,
            )
            fresh_clean = np.asarray(
                fresh_problem.action_fn(fresh_problem.memory), dtype=np.float32
            )[..., : args.robot_action_dim]
            fresh_degraded = np.asarray(
                fresh_problem.action_fn(jnp.asarray(degraded_np, dtype=fresh_problem.memory.dtype)),
                dtype=np.float32,
            )[..., : args.robot_action_dim]
            fresh_edited = np.asarray(
                fresh_problem.action_fn(
                    jnp.asarray(pullback.quantized_memory, dtype=fresh_problem.memory.dtype)
                ),
                dtype=np.float32,
            )[..., : args.robot_action_dim]
            fresh_recoveries.append(_recovery(fresh_clean, fresh_degraded, fresh_edited))

        reports.append(
            {
                "probe_radius": probe_radius,
                "response_rank": rank,
                "linear_target_residual_before": float(np.linalg.norm(target_delta)),
                "linear_target_residual_after": pullback.linear_residual_norm,
                "applied_relative_delta_norm": pullback.applied_relative_norm,
                "same_noise_recovery": _recovery(
                    clean_actions, degraded_actions, edited_actions
                ),
                "negative_direction_recovery": _recovery(
                    clean_actions, degraded_actions, negative_actions
                ),
                "random_direction_recovery": _recovery(
                    clean_actions, degraded_actions, random_actions
                ),
                "fresh_noise_recoveries": fresh_recoveries,
                "fresh_noise_mean_recovery": float(np.mean(fresh_recoveries)),
                "secant_columns": [asdict(item) for item in response.diagnostics],
                "singular_values": pullback.singular_values.tolist(),
            }
        )
        key = f"r{probe_radius:g}".replace(".", "p").replace("-", "m")
        artifact_arrays[f"edited_memory_{key}"] = pullback.quantized_memory
        artifact_arrays[f"applied_delta_{key}"] = pullback.applied_delta
        artifact_arrays[f"response_matrix_{key}"] = response.response_matrix
        artifact_arrays[f"edited_actions_{key}"] = edited_actions

    report = {
        "experiment": "E12S_deployed_secant_memory_recovery",
        "checkpoint": str(args.checkpoint),
        "data": str(args.data),
        "index": args.index,
        "preset": args.preset,
        "basis": args.basis,
        "basis_directions": int(basis.shape[0]),
        "drop_fraction": args.drop_fraction,
        "dropped_history_rows": dropped,
        "max_update_radius": args.max_update_radius,
        "robot_action_dim": args.robot_action_dim,
        "radius_results": reports,
        "claim_scope": (
            "reward-free action recovery using deployed BF16 finite responses; "
            "not environment success and not an AD-JVP result"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    artifact_path = args.artifacts or args.output.with_suffix(".npz")
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(artifact_path, **artifact_arrays)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
