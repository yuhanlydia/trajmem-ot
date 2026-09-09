#!/usr/bin/env python3
"""E12-S2: reward-free BF16 finite-response recovery with matched controls."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from trajmem_ot.finite_response import (
    QuantizedMatchedMemory,
    build_finite_response_model,
    quantized_norm_matched_memory,
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
    parser.add_argument("--corruption-seed", type=int)
    parser.add_argument(
        "--corruption-mode",
        choices=("random_rows", "contiguous_rows"),
        default="random_rows",
    )
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
    parser.add_argument("--random-controls", type=int, default=16)
    parser.add_argument("--control-norm-tolerance", type=float, default=0.03)
    parser.add_argument("--robot-action-dim", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path)
    return parser.parse_args()


def _degrade(
    memory: np.ndarray,
    valid_mask: np.ndarray | None,
    fraction: float,
    seed: int,
    mode: str,
) -> tuple[np.ndarray, list[int]]:
    if not 0.0 < fraction < 1.0:
        raise ValueError("drop-fraction must be in (0, 1)")
    valid = np.arange(memory.shape[0]) if valid_mask is None else np.flatnonzero(valid_mask)
    if valid.size == 0:
        raise ValueError("history mask contains no valid rows")
    count = max(1, min(valid.size, int(round(valid.size * fraction))))
    rng = np.random.default_rng(seed)
    if mode == "random_rows":
        dropped = np.sort(rng.choice(valid, size=count, replace=False))
    elif mode == "contiguous_rows":
        start = int(rng.integers(0, valid.size - count + 1))
        dropped = valid[start : start + count]
    else:  # pragma: no cover - argparse protects this path
        raise ValueError(f"unsupported corruption mode: {mode}")
    degraded = memory.copy()
    degraded[dropped] = 0.0
    return degraded, dropped.astype(int).tolist()


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
    return normalize_basis(
        np.concatenate([direction[None, ...], random[: count - 1]], axis=0)
    )


def _recovery(clean: np.ndarray, degraded: np.ndarray, candidate: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(degraded - clean)), 1e-12)
    return float(1.0 - np.linalg.norm(candidate - clean) / denominator)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 1e-20 else 0.0


def _matched_control(
    memory: Any,
    direction: np.ndarray,
    *,
    target_norm: float,
    tolerance: float,
) -> QuantizedMatchedMemory:
    result = quantized_norm_matched_memory(
        memory,
        direction,
        target_norm=target_norm,
        relative_tolerance=tolerance,
    )
    if result.relative_error > tolerance:
        raise RuntimeError(
            "unable to match the deployed BF16 control norm: "
            f"relative_error={result.relative_error:.6f} > {tolerance:.6f}"
        )
    return result


def _control_record(
    matched: QuantizedMatchedMemory,
    *,
    same_noise_recovery: float,
    fresh_recoveries: list[float],
) -> dict[str, Any]:
    return {
        "applied_norm": matched.applied_norm,
        "target_norm": matched.target_norm,
        "relative_norm_error": matched.relative_error,
        "scale": matched.scale,
        "changed_fraction": matched.changed_fraction,
        "same_noise_recovery": same_noise_recovery,
        "fresh_noise_recoveries": fresh_recoveries,
        "fresh_noise_mean_recovery": float(np.mean(fresh_recoveries)),
    }


def main() -> None:
    args = parse_args()
    if args.fresh_noises <= 0 or args.random_controls <= 0:
        raise ValueError("fresh-noises and random-controls must be positive")
    if not 0.0 < args.control_norm_tolerance < 1.0:
        raise ValueError("control-norm-tolerance must be in (0, 1)")

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
    corruption_seed = (
        int(args.corruption_seed)
        if args.corruption_seed is not None
        else int(args.seed + args.index)
    )
    degraded_np, dropped = _degrade(
        clean_np,
        mask,
        args.drop_fraction,
        corruption_seed,
        args.corruption_mode,
    )
    degraded = jnp.asarray(degraded_np, dtype=problem.memory.dtype)
    clean_actions = np.asarray(problem.action_fn(problem.memory), dtype=np.float32)[
        ..., : args.robot_action_dim
    ]
    degraded_actions = np.asarray(problem.action_fn(degraded), dtype=np.float32)[
        ..., : args.robot_action_dim
    ]
    target_delta = clean_actions - degraded_actions
    basis = _load_basis(args, degraded_np, clean_np - degraded_np, basis_count)

    reports: list[dict[str, Any]] = []
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
        target_control_norm = float(np.linalg.norm(pullback.applied_delta))
        if target_control_norm <= 1e-12:
            raise RuntimeError("finite-response pullback was erased by BF16 quantization")
        negative_match = _matched_control(
            degraded,
            -pullback.applied_delta,
            target_norm=target_control_norm,
            tolerance=args.control_norm_tolerance,
        )

        random_matches: list[QuantizedMatchedMemory] = []
        rng = np.random.default_rng(args.seed + 900 + 10_000 * radius_index)
        for _ in range(args.random_controls):
            random_matches.append(
                _matched_control(
                    degraded,
                    rng.normal(size=degraded_np.shape).astype(np.float32),
                    target_norm=target_control_norm,
                    tolerance=args.control_norm_tolerance,
                )
            )

        edited_actions = np.asarray(problem.action_fn(edited), dtype=np.float32)[
            ..., : args.robot_action_dim
        ]
        negative_actions = np.asarray(
            problem.action_fn(
                jnp.asarray(negative_match.quantized_memory, dtype=problem.memory.dtype)
            ),
            dtype=np.float32,
        )[..., : args.robot_action_dim]
        random_actions = [
            np.asarray(
                problem.action_fn(
                    jnp.asarray(match.quantized_memory, dtype=problem.memory.dtype)
                ),
                dtype=np.float32,
            )[..., : args.robot_action_dim]
            for match in random_matches
        ]

        same_recovery = _recovery(clean_actions, degraded_actions, edited_actions)
        negative_same = _recovery(clean_actions, degraded_actions, negative_actions)
        random_same = [
            _recovery(clean_actions, degraded_actions, action)
            for action in random_actions
        ]

        fresh_ours: list[float] = []
        fresh_negative: list[float] = []
        fresh_random: list[list[float]] = [
            [] for _ in range(args.random_controls)
        ]
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
                fresh_problem.action_fn(
                    jnp.asarray(degraded_np, dtype=fresh_problem.memory.dtype)
                ),
                dtype=np.float32,
            )[..., : args.robot_action_dim]
            fresh_edited = np.asarray(
                fresh_problem.action_fn(
                    jnp.asarray(
                        pullback.quantized_memory, dtype=fresh_problem.memory.dtype
                    )
                ),
                dtype=np.float32,
            )[..., : args.robot_action_dim]
            fresh_negative_actions = np.asarray(
                fresh_problem.action_fn(
                    jnp.asarray(
                        negative_match.quantized_memory,
                        dtype=fresh_problem.memory.dtype,
                    )
                ),
                dtype=np.float32,
            )[..., : args.robot_action_dim]
            fresh_ours.append(_recovery(fresh_clean, fresh_degraded, fresh_edited))
            fresh_negative.append(
                _recovery(fresh_clean, fresh_degraded, fresh_negative_actions)
            )
            for control_index, match in enumerate(random_matches):
                fresh_random_action = np.asarray(
                    fresh_problem.action_fn(
                        jnp.asarray(
                            match.quantized_memory,
                            dtype=fresh_problem.memory.dtype,
                        )
                    ),
                    dtype=np.float32,
                )[..., : args.robot_action_dim]
                fresh_random[control_index].append(
                    _recovery(fresh_clean, fresh_degraded, fresh_random_action)
                )

        actual_delta = edited_actions - degraded_actions
        predicted_delta = pullback.linear_prediction.reshape(actual_delta.shape)
        actual_response_norm = float(np.linalg.norm(actual_delta))
        predicted_response_norm = float(np.linalg.norm(predicted_delta))
        target_norm = float(np.linalg.norm(target_delta))
        random_records = [
            _control_record(
                match,
                same_noise_recovery=same,
                fresh_recoveries=fresh,
            )
            for match, same, fresh in zip(
                random_matches, random_same, fresh_random, strict=True
            )
        ]
        random_fresh_means = [
            float(record["fresh_noise_mean_recovery"])
            for record in random_records
        ]
        row = {
            "probe_radius": probe_radius,
            "response_rank": rank,
            "linear_target_residual_before": target_norm,
            "linear_target_residual_after": pullback.linear_residual_norm,
            "actual_target_residual_after": float(
                np.linalg.norm(actual_delta - target_delta)
            ),
            "applied_relative_delta_norm": pullback.applied_relative_norm,
            "same_noise_recovery": same_recovery,
            "fresh_noise_recoveries": fresh_ours,
            "fresh_noise_mean_recovery": float(np.mean(fresh_ours)),
            "response_fidelity": {
                "actual_vs_linear_cosine": _cosine(actual_delta, predicted_delta),
                "relative_error_to_actual": float(
                    np.linalg.norm(actual_delta - predicted_delta)
                    / max(actual_response_norm, 1e-12)
                ),
                "actual_response_norm": actual_response_norm,
                "linear_prediction_norm": predicted_response_norm,
                "actual_to_predicted_norm_ratio": actual_response_norm
                / max(predicted_response_norm, 1e-12),
            },
            "negative_control": _control_record(
                negative_match,
                same_noise_recovery=negative_same,
                fresh_recoveries=fresh_negative,
            ),
            "random_controls": random_records,
            "negative_direction_recovery": negative_same,
            "random_direction_recovery": float(np.median(random_same)),
            "fresh_noise_negative_mean_recovery": float(np.mean(fresh_negative)),
            "fresh_noise_random_median_recovery": float(
                np.median(random_fresh_means)
            ),
            "fresh_noise_random_all_mean_recoveries": random_fresh_means,
            "fresh_noise_ours_minus_random_median": float(
                np.mean(fresh_ours) - np.median(random_fresh_means)
            ),
            "secant_columns": [asdict(item) for item in response.diagnostics],
            "singular_values": pullback.singular_values.tolist(),
        }
        reports.append(row)

        key = f"r{probe_radius:g}".replace(".", "p").replace("-", "m")
        artifact_arrays[f"edited_memory_{key}"] = pullback.quantized_memory
        artifact_arrays[f"applied_delta_{key}"] = pullback.applied_delta
        artifact_arrays[f"negative_memory_{key}"] = negative_match.quantized_memory
        artifact_arrays[f"response_matrix_{key}"] = response.response_matrix
        artifact_arrays[f"edited_actions_{key}"] = edited_actions
        artifact_arrays[f"linear_prediction_{key}"] = predicted_delta
        artifact_arrays[f"actual_response_{key}"] = actual_delta
        artifact_arrays[f"random_control_applied_norms_{key}"] = np.asarray(
            [match.applied_norm for match in random_matches], dtype=np.float32
        )

    report = {
        "experiment": "E12S2_deployed_secant_memory_recovery",
        "checkpoint": str(args.checkpoint),
        "data": str(args.data),
        "index": args.index,
        "preset": args.preset,
        "basis": args.basis,
        "basis_directions": int(basis.shape[0]),
        "corruption_mode": args.corruption_mode,
        "corruption_seed": corruption_seed,
        "drop_fraction": args.drop_fraction,
        "dropped_history_rows": dropped,
        "max_update_radius": args.max_update_radius,
        "random_controls": args.random_controls,
        "control_norm_tolerance": args.control_norm_tolerance,
        "robot_action_dim": args.robot_action_dim,
        "radius_results": reports,
        "claim_scope": (
            "reward-free action recovery using deployed BF16 finite responses "
            "with applied-norm-matched controls; not environment success"
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
