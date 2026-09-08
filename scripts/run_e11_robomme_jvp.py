#!/usr/bin/env python3
"""E11: exact JAX memory-to-action JVPs on one released RoboMME state."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from trajmem_ot.jax_operator import (
    action_jvp,
    batched_action_jvps,
    energy_rank,
    quantized_action_chord,
    response_svd,
)
from trajmem_ot.memory_basis import random_rank_one_basis
from trajmem_ot.presets import get_compute_preset
from trajmem_ot.robomme_jax import build_fixed_noise_action_problem
from trajmem_ot.robomme_runtime import load_runtime_state


def _parse_float_list(value: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected a comma-separated list of positive numbers")
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
    parser.add_argument("--fd-directions", type=int, default=2)
    parser.add_argument(
        "--fd-radii",
        type=_parse_float_list,
        default=_parse_float_list("3.125e-5,6.25e-5,1.25e-4,2.5e-4"),
        help="relative Frobenius memory radii used only to validate exact JVPs",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path)
    return parser.parse_args()


def _block(tree):
    import jax

    return jax.tree.map(lambda value: value.block_until_ready(), tree)


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
    noise_key, model_key = jax.random.split(jax.random.key(args.seed + 101))
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
    memory_np = np.asarray(problem.memory, dtype=np.float32)
    basis_np = random_rank_one_basis(
        memory_np.shape,
        count=preset.basis_directions,
        seed=args.seed + args.index * 1009,
    )
    basis = jnp.asarray(basis_np)

    started = time.perf_counter()
    base_actions, responses = batched_action_jvps(
        problem.action_fn,
        problem.memory,
        basis,
        chunk_size=preset.jvp_chunk_size,
    )
    _block((base_actions, responses))
    first_latency = time.perf_counter() - started

    started = time.perf_counter()
    _, responses_warm = batched_action_jvps(
        problem.action_fn,
        problem.memory,
        basis,
        chunk_size=preset.jvp_chunk_size,
    )
    _block(responses_warm)
    warm_latency = time.perf_counter() - started

    base_np = np.asarray(base_actions)
    responses_np = np.asarray(responses)
    response_matrix = responses_np.reshape(responses_np.shape[0], -1).T
    _, singular_values, _ = response_svd(response_matrix)
    memory_norm = float(np.linalg.norm(memory_np))

    comparisons = []
    memory_jax = jnp.asarray(problem.memory)
    for direction_index in range(min(args.fd_directions, basis_np.shape[0])):
        exact = response_matrix[:, direction_index]
        for radius in args.fd_radii:
            step = radius * memory_norm
            action_plus, action_minus, actual_half, memory_plus, memory_minus = (
                quantized_action_chord(
                    problem.action_fn,
                    memory_jax,
                    basis[direction_index],
                    step=step,
                )
            )
            _, actual_jvp = action_jvp(problem.action_fn, memory_jax, actual_half)
            _block((action_plus, action_minus, actual_half, memory_plus, memory_minus, actual_jvp))
            plus_np = np.asarray(memory_plus)
            minus_np = np.asarray(memory_minus)
            actual_half_np = np.asarray(actual_half)
            # Perform output subtraction in FP32 even when a model emits BF16.
            # Otherwise the chord itself can be dominated by output quantization.
            plus_action_np = np.asarray(action_plus, dtype=np.float32)
            minus_action_np = np.asarray(action_minus, dtype=np.float32)
            chord = ((plus_action_np - minus_action_np) / 2.0).reshape(-1)
            actual_jvp_np = np.asarray(actual_jvp, dtype=np.float32).reshape(-1)
            changed_plus = float(np.mean(plus_np != memory_np))
            changed_minus = float(np.mean(minus_np != memory_np))
            actual_radius = float(np.linalg.norm(actual_half_np) / max(memory_norm, 1e-12))
            asymmetry = float(
                np.linalg.norm((plus_np - memory_np) + (minus_np - memory_np))
                / max(np.linalg.norm(plus_np - minus_np), 1e-12)
            )
            denominator = max(np.linalg.norm(actual_jvp_np) * np.linalg.norm(chord), 1e-12)
            comparisons.append(
                {
                    "direction": direction_index,
                    "nominal_relative_radius": radius,
                    "absolute_step": step,
                    "cosine": float(actual_jvp_np @ chord / denominator),
                    "relative_error": float(
                        np.linalg.norm(actual_jvp_np - chord) / max(np.linalg.norm(chord), 1e-12)
                    ),
                    "exact_response_norm": float(np.linalg.norm(exact)),
                    "actual_jvp_norm": float(np.linalg.norm(actual_jvp_np)),
                    "action_chord_norm": float(np.linalg.norm(chord)),
                    "memory_dtype": str(memory_jax.dtype),
                    "actual_half_dtype": str(actual_half.dtype),
                    "action_plus_dtype": str(action_plus.dtype),
                    "action_minus_dtype": str(action_minus.dtype),
                    "jvp_dtype": str(actual_jvp.dtype),
                    "changed_fraction_plus": changed_plus,
                    "changed_fraction_minus": changed_minus,
                    "actual_relative_radius": actual_radius,
                    "asymmetry": asymmetry,
                    "qualified": bool(min(changed_plus, changed_minus) > 0.005 and asymmetry < 0.25),
                }
            )

    report = {
        "experiment": "E11_exact_jvp",
        "checkpoint": str(args.checkpoint),
        "data": str(args.data),
        "index": args.index,
        "preset": preset.name,
        "device": [str(device) for device in jax.devices()],
        "history_config": runtime.history_config_name,
        "memory_field": args.memory_field,
        "memory_shape": list(memory_np.shape),
        "memory_norm": memory_norm,
        "action_shape": list(base_np.shape),
        "basis_directions": int(basis_np.shape[0]),
        "jvp_chunk_size": preset.jvp_chunk_size,
        "first_call_latency_s": first_latency,
        "warm_call_latency_s": warm_latency,
        "singular_values": singular_values.tolist(),
        "effective_rank_90": energy_rank(singular_values, threshold=0.9),
        "fd_comparisons": comparisons,
        "min_fd_cosine": min(row["cosine"] for row in comparisons) if comparisons else None,
        "max_fd_relative_error": max(row["relative_error"] for row in comparisons) if comparisons else None,
        "claim_scope": "operator correctness and local memory-to-action controllability only",
        "finite_difference_protocol": "quantization_aware_actual_chord",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    artifact_path = args.artifacts or args.output.with_suffix(".npz")
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        artifact_path,
        memory=memory_np,
        basis=basis_np,
        base_actions=base_np,
        responses=responses_np,
        response_matrix=response_matrix,
        demonstration_actions_physical=np.asarray(runtime.item.get("actions", [])),
        noise=np.asarray(noise),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
