#!/usr/bin/env python3
"""CPU-safe end-to-end smoke test for the E11 JAX response operator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from trajmem_ot.jax_operator import (
    batched_action_jvps,
    central_action_secant,
    energy_rank,
    svd_ridge_pullback,
)
from trajmem_ot.memory_basis import random_rank_one_basis
from trajmem_ot.presets import get_compute_preset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("16gb", "24gb"), default="16gb")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preset = get_compute_preset(args.preset)
    rng = np.random.default_rng(args.seed)
    memory_np = rng.normal(scale=0.25, size=(12, 8)).astype(np.float32)
    hidden_weights = rng.normal(scale=0.2, size=(96, 32)).astype(np.float32)
    output_weights = rng.normal(scale=0.2, size=(32, 24)).astype(np.float32)
    memory = jnp.asarray(memory_np)
    hidden_weights_jax = jnp.asarray(hidden_weights)
    output_weights_jax = jnp.asarray(output_weights)

    @jax.jit
    def action_fn(candidate_memory):
        hidden = jnp.tanh(candidate_memory.reshape(-1) @ hidden_weights_jax)
        return (hidden @ output_weights_jax).reshape(6, 4)

    basis_np = random_rank_one_basis(
        memory_np.shape, count=preset.basis_directions, seed=args.seed + 1
    )
    basis = jnp.asarray(basis_np)

    start = time.perf_counter()
    base_actions, responses = batched_action_jvps(
        action_fn, memory, basis, chunk_size=preset.jvp_chunk_size
    )
    jax.block_until_ready((base_actions, responses))
    first_latency = time.perf_counter() - start

    start = time.perf_counter()
    _, warm_responses = batched_action_jvps(
        action_fn, memory, basis, chunk_size=preset.jvp_chunk_size
    )
    jax.block_until_ready(warm_responses)
    warm_latency = time.perf_counter() - start

    response_np = np.asarray(responses).reshape(preset.basis_directions, -1).T
    secant_cosines = []
    secant_relative_errors = []
    for index in range(min(4, preset.basis_directions)):
        tangent = response_np[:, index]
        secant = np.asarray(
            central_action_secant(action_fn, memory, basis[index], step=1e-3)
        ).reshape(-1)
        denom = max(np.linalg.norm(tangent) * np.linalg.norm(secant), 1e-12)
        secant_cosines.append(float(tangent @ secant / denom))
        secant_relative_errors.append(
            float(np.linalg.norm(tangent - secant) / max(np.linalg.norm(tangent), 1e-12))
        )

    true_coefficients = np.zeros(preset.basis_directions, dtype=np.float64)
    active = min(3, preset.response_rank)
    true_coefficients[:active] = np.linspace(0.5, 1.0, active)
    target = response_np @ true_coefficients
    pullback = svd_ridge_pullback(
        response_np,
        target,
        damping=1e-6,
        rank=preset.response_rank,
    )
    report = {
        "preset": preset.name,
        "device": str(jax.devices()[0]),
        "basis_directions": preset.basis_directions,
        "jvp_chunk_size": preset.jvp_chunk_size,
        "response_rank_requested": preset.response_rank,
        "effective_rank_90": energy_rank(pullback.singular_values, threshold=0.9),
        "jvp_secant_min_cosine": min(secant_cosines),
        "jvp_secant_max_relative_error": max(secant_relative_errors),
        "first_call_latency_s": first_latency,
        "warm_call_latency_s": warm_latency,
        "pullback_residual_before": float(np.linalg.norm(target)),
        "pullback_residual_after": pullback.residual_norm,
        "action_shape": list(np.shape(base_actions)),
        "response_shape": list(response_np.shape),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
