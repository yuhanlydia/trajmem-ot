#!/usr/bin/env python3
"""E14-A: pull a return-tilted trajectory OT target through BF16 finite responses."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from trajmem_ot.finite_response import (
    build_finite_response_model,
    quantized_norm_matched_memory,
    solve_finite_response_pullback,
)
from trajmem_ot.memory_basis import normalize_basis, random_rank_one_basis
from trajmem_ot.robomme_jax import build_fixed_noise_action_problem
from trajmem_ot.robomme_runtime import load_runtime_state
from trajmem_ot.transport import return_tilted_transport


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--index", type=int)
    parser.add_argument("--preset", choices=("16gb", "24gb"), default="16gb")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--config", default="mme_vla_suite")
    parser.add_argument("--history-config")
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--basis", choices=("random", "history"), default="history")
    parser.add_argument("--history-basis", type=Path)
    parser.add_argument("--probe-radius", type=float, default=2.5e-4)
    parser.add_argument("--max-update-radius", type=float, default=2.5e-3)
    parser.add_argument("--damping", type=float, default=1e-4)
    parser.add_argument("--response-rank", type=int)
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument("--sinkhorn-epsilon", type=float)
    parser.add_argument("--sinkhorn-iters", type=int, default=100)
    parser.add_argument("--velocity-weight", type=float, default=0.25)
    parser.add_argument("--random-controls", type=int, default=8)
    parser.add_argument("--control-norm-tolerance", type=float, default=0.03)
    parser.add_argument("--robot-action-dim", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path)
    return parser.parse_args()


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 1e-20 else 0.0


def _target_fit(target: np.ndarray, actual: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(target)), 1e-12)
    return float(1.0 - np.linalg.norm(actual - target) / denominator)


def _load_manifest(
    path: Path, explicit_index: int | None
) -> tuple[int, list[int], np.ndarray, dict[str, Any]]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("manifest must be a JSON object")
    index = explicit_index if explicit_index is not None else payload.get("index")
    if index is None or int(index) < 0:
        raise ValueError("manifest or --index must provide a non-negative dataset index")
    particles = payload.get("particles")
    if not isinstance(particles, list) or len(particles) < 2:
        raise ValueError("manifest particles must contain at least two entries")
    seeds = [int(row["noise_seed"]) for row in particles]
    if len(set(seeds)) != len(seeds):
        raise ValueError("manifest noise_seed values must be unique")
    returns = np.asarray([float(row["return"]) for row in particles], dtype=np.float64)
    if not np.isfinite(returns).all():
        raise ValueError("manifest returns must be finite")
    if float(returns.std()) <= 1e-8:
        raise ValueError("manifest returns contain no ranking signal")
    return int(index), seeds, returns, payload


def _load_basis(
    args: argparse.Namespace, memory_shape: tuple[int, ...], count: int
) -> np.ndarray:
    if args.basis == "random":
        return random_rank_one_basis(memory_shape, count=count, seed=args.seed + 511)
    if args.history_basis is None:
        raise ValueError("--history-basis is required for --basis=history")
    basis = normalize_basis(np.asarray(np.load(args.history_basis)["basis"]))
    if basis.shape[1:] != memory_shape:
        raise ValueError(f"basis shape {basis.shape[1:]} != memory shape {memory_shape}")
    return basis[: min(count, basis.shape[0])]


def main() -> None:
    args = parse_args()
    if args.random_controls <= 0:
        raise ValueError("random-controls must be positive")
    index, noise_seeds, returns, manifest = _load_manifest(args.manifest, args.index)
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
        index=index,
        seed=args.seed,
        train_config_name=args.config,
        history_config_name=args.history_config,
    )
    model = runtime.policy._model
    problems = []
    for noise_seed in noise_seeds:
        noise_key, model_key = jax.random.split(jax.random.key(noise_seed))
        noise = jax.random.normal(
            noise_key,
            (1, int(model.action_horizon), int(model.action_dim)),
            dtype=jnp.float32,
        )
        problems.append(
            build_fixed_noise_action_problem(
                runtime.policy,
                runtime.observation,
                noise=noise,
                rng=model_key,
                memory_field="static_image_emb",
                num_steps=args.num_steps,
            )
        )
    memory = problems[0].memory
    if any(tuple(problem.memory.shape) != tuple(memory.shape) for problem in problems):
        raise RuntimeError("particle problems disagree on memory shape")

    def stacked_action_fn(candidate_memory):
        return jnp.stack([problem.action_fn(candidate_memory) for problem in problems], axis=0)

    base_actions_all = np.asarray(stacked_action_fn(memory), dtype=np.float32)
    base_actions = base_actions_all[..., : args.robot_action_dim]
    transport = return_tilted_transport(
        base_actions,
        returns,
        beta=args.beta,
        epsilon=args.sinkhorn_epsilon,
        sinkhorn_iterations=args.sinkhorn_iters,
        velocity_weight=args.velocity_weight,
    )
    target_delta = transport.transport.astype(np.float32)
    basis = _load_basis(args, tuple(memory.shape), basis_count)
    response = build_finite_response_model(
        stacked_action_fn,
        memory,
        basis,
        relative_radius=args.probe_radius,
        robot_action_dim=args.robot_action_dim,
    )
    rank = min(args.response_rank or default_rank, min(response.response_matrix.shape))
    pullback = solve_finite_response_pullback(
        response,
        memory,
        target_delta,
        damping=args.damping,
        rank=rank,
        max_relative_norm=args.max_update_radius,
    )

    edited = jnp.asarray(pullback.quantized_memory, dtype=memory.dtype)
    edited_actions = np.asarray(stacked_action_fn(edited), dtype=np.float32)[
        ..., : args.robot_action_dim
    ]
    actual_delta = edited_actions - base_actions
    target_norm = float(np.linalg.norm(target_delta))
    applied_norm = float(np.linalg.norm(pullback.applied_delta))
    if applied_norm <= 1e-12:
        raise RuntimeError("OT pullback was erased by the deployed memory dtype")

    negative_match = quantized_norm_matched_memory(
        memory,
        -pullback.applied_delta,
        target_norm=applied_norm,
        relative_tolerance=args.control_norm_tolerance,
    )
    if negative_match.relative_error > args.control_norm_tolerance:
        raise RuntimeError("negative control could not be matched after quantization")
    negative_actions = np.asarray(
        stacked_action_fn(jnp.asarray(negative_match.quantized_memory, dtype=memory.dtype)),
        dtype=np.float32,
    )[..., : args.robot_action_dim]
    negative_delta = negative_actions - base_actions

    rng = np.random.default_rng(args.seed + 8_000)
    random_rows = []
    random_arrays = []
    for control_index in range(args.random_controls):
        match = quantized_norm_matched_memory(
            memory,
            rng.normal(size=np.shape(memory)).astype(np.float32),
            target_norm=applied_norm,
            relative_tolerance=args.control_norm_tolerance,
        )
        if match.relative_error > args.control_norm_tolerance:
            raise RuntimeError(
                f"random control {control_index} could not be matched after quantization"
            )
        actions = np.asarray(
            stacked_action_fn(jnp.asarray(match.quantized_memory, dtype=memory.dtype)),
            dtype=np.float32,
        )[..., : args.robot_action_dim]
        delta = actions - base_actions
        random_rows.append(
            {
                "control_index": control_index,
                "applied_norm": match.applied_norm,
                "relative_norm_error": match.relative_error,
                "target_fit": _target_fit(target_delta, delta),
                "target_alignment_cosine": _cosine(target_delta, delta),
            }
        )
        random_arrays.append(delta)

    report = {
        "experiment": "E14A_return_tilted_ot_finite_response_pullback",
        "checkpoint": str(args.checkpoint),
        "data": str(args.data),
        "index": index,
        "manifest": str(args.manifest),
        "manifest_metadata": {
            key: value for key, value in manifest.items() if key != "particles"
        },
        "noise_seeds": noise_seeds,
        "returns": returns.tolist(),
        "preset": args.preset,
        "basis": args.basis,
        "basis_directions": int(basis.shape[0]),
        "probe_radius": args.probe_radius,
        "max_update_radius": args.max_update_radius,
        "response_rank": rank,
        "applied_relative_delta_norm": pullback.applied_relative_norm,
        "target_transport_norm": target_norm,
        "linear_target_residual_before": target_norm,
        "linear_target_residual_after": pullback.linear_residual_norm,
        "actual_target_fit": _target_fit(target_delta, actual_delta),
        "actual_target_alignment_cosine": _cosine(target_delta, actual_delta),
        "actual_target_residual": float(np.linalg.norm(actual_delta - target_delta)),
        "negative_control": {
            "applied_norm": negative_match.applied_norm,
            "relative_norm_error": negative_match.relative_error,
            "target_fit": _target_fit(target_delta, negative_delta),
            "target_alignment_cosine": _cosine(target_delta, negative_delta),
        },
        "random_controls": random_rows,
        "target_weights": transport.target_weights.tolist(),
        "source_weights": transport.source_weights.tolist(),
        "sinkhorn_epsilon": transport.epsilon,
        "secant_columns": [asdict(item) for item in response.diagnostics],
        "singular_values": pullback.singular_values.tolist(),
        "claim_scope": (
            "open-loop return-conditioned action-transport fit through a deployed "
            "BF16 finite-response operator; environment improvement requires paired "
            "simulator rollout of the saved actions"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    artifact_path = args.artifacts or args.output.with_suffix(".npz")
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        artifact_path,
        base_memory=np.asarray(memory, dtype=np.float32),
        edited_memory=pullback.quantized_memory,
        applied_delta=pullback.applied_delta,
        base_actions=base_actions,
        edited_actions=edited_actions,
        negative_actions=negative_actions,
        random_action_deltas=np.stack(random_arrays),
        returns=returns,
        source_weights=transport.source_weights,
        target_weights=transport.target_weights,
        coupling=transport.coupling,
        target_transport=target_delta,
        actual_action_delta=actual_delta,
        response_matrix=response.response_matrix,
        noise_seeds=np.asarray(noise_seeds, dtype=np.int64),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
