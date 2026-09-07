#!/usr/bin/env python3
"""E12: reward-free clean/degraded-memory recovery with JVP-SVD pullback."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from trajmem_ot.jax_operator import batched_action_jvps, energy_rank, svd_ridge_pullback
from trajmem_ot.memory_basis import combine_basis, normalize_basis, random_rank_one_basis
from trajmem_ot.presets import get_compute_preset
from trajmem_ot.robomme_jax import build_fixed_noise_action_problem
from trajmem_ot.robomme_runtime import load_runtime_state


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
    parser.add_argument("--drop-fraction", type=float, default=0.20)
    parser.add_argument("--trust-radius", type=float, default=2.5e-4)
    parser.add_argument("--damping", type=float, default=1e-3)
    parser.add_argument("--basis", choices=("random", "history", "hybrid"), default="hybrid")
    parser.add_argument("--history-basis", type=Path)
    parser.add_argument("--fresh-noises", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path)
    return parser.parse_args()


def _load_basis(args, memory: np.ndarray, corruption_direction: np.ndarray, count: int) -> np.ndarray:
    random = random_rank_one_basis(memory.shape, count=count, seed=args.seed + 211)
    if args.basis == "random":
        return random
    if args.basis == "history":
        if args.history_basis is None:
            raise ValueError("--history-basis is required when --basis=history")
        loaded = np.load(args.history_basis)
        history = normalize_basis(np.asarray(loaded["basis"]))
        if history.shape[1:] != memory.shape:
            raise ValueError(f"history basis shape {history.shape[1:]} != memory shape {memory.shape}")
        return history[: min(count, history.shape[0])]
    corruption = corruption_direction / max(np.linalg.norm(corruption_direction), 1e-12)
    if np.linalg.norm(corruption) <= 1e-12:
        raise ValueError("memory degradation produced no nonzero direction")
    return normalize_basis(np.concatenate([corruption[None, ...], random[: count - 1]], axis=0))


def _clip_delta(memory: np.ndarray, delta: np.ndarray, relative_radius: float) -> np.ndarray:
    maximum = relative_radius * np.linalg.norm(memory)
    norm = np.linalg.norm(delta)
    if norm <= maximum or norm <= 1e-12:
        return delta
    return delta * (maximum / norm)


def _degrade(memory: np.ndarray, valid_mask: np.ndarray | None, fraction: float, seed: int) -> tuple[np.ndarray, list[int]]:
    if not 0.0 < fraction < 1.0:
        raise ValueError("drop-fraction must be in (0, 1)")
    if memory.ndim != 2:
        raise ValueError("E12 currently supports 2-D perceptual memory only")
    valid = np.arange(memory.shape[0]) if valid_mask is None else np.flatnonzero(valid_mask)
    if valid.size == 0:
        raise ValueError("memory mask contains no valid history tokens")
    count = max(1, int(round(valid.size * fraction)))
    rng = np.random.default_rng(seed)
    dropped = np.sort(rng.choice(valid, size=min(count, valid.size), replace=False))
    degraded = memory.copy()
    degraded[dropped] = 0.0
    return degraded, dropped.tolist()


def _recovery(clean: np.ndarray, degraded: np.ndarray, candidate: np.ndarray) -> float:
    denominator = max(np.linalg.norm(degraded - clean), 1e-12)
    return float(1.0 - np.linalg.norm(candidate - clean) / denominator)


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
        memory_field=args.memory_field,
        num_steps=args.num_steps,
    )
    clean_memory = np.asarray(problem.memory, dtype=np.float32)
    mask_value = getattr(runtime.observation, "static_mask", None) if args.memory_field == "static_image_emb" else getattr(runtime.observation, "recur_mask", None)
    valid_mask = None if mask_value is None else np.asarray(mask_value[0], dtype=bool)
    degraded_memory, dropped = _degrade(clean_memory, valid_mask, args.drop_fraction, args.seed + args.index)
    clean_actions = np.asarray(problem.action_fn(jnp.asarray(clean_memory)))
    degraded_actions = np.asarray(problem.action_fn(jnp.asarray(degraded_memory)))
    target = (clean_actions - degraded_actions).reshape(-1)

    basis_np = _load_basis(
        args,
        degraded_memory,
        clean_memory - degraded_memory,
        preset.basis_directions,
    )
    _, responses = batched_action_jvps(
        problem.action_fn,
        jnp.asarray(degraded_memory),
        jnp.asarray(basis_np),
        chunk_size=min(preset.jvp_chunk_size, basis_np.shape[0]),
    )
    responses_np = np.asarray(responses)
    response_matrix = responses_np.reshape(responses_np.shape[0], -1).T
    rank = min(preset.response_rank, min(response_matrix.shape))
    pullback = svd_ridge_pullback(
        response_matrix,
        target,
        damping=args.damping,
        rank=rank,
    )
    delta = combine_basis(basis_np, pullback.coefficients)
    delta = _clip_delta(degraded_memory, delta, args.trust_radius)
    edited_memory = degraded_memory + delta
    negative_memory = degraded_memory - delta
    rng = np.random.default_rng(args.seed + 401)
    random_delta = rng.normal(size=delta.shape).astype(np.float32)
    random_delta *= np.linalg.norm(delta) / max(np.linalg.norm(random_delta), 1e-12)
    random_memory = degraded_memory + random_delta

    edited_actions = np.asarray(problem.action_fn(jnp.asarray(edited_memory)))
    negative_actions = np.asarray(problem.action_fn(jnp.asarray(negative_memory)))
    random_actions = np.asarray(problem.action_fn(jnp.asarray(random_memory)))

    fresh = []
    for offset in range(args.fresh_noises):
        fresh_noise_key, fresh_model_key = jax.random.split(jax.random.key(args.seed + 1000 + offset))
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
            memory_field=args.memory_field,
            num_steps=args.num_steps,
        )
        fresh_clean = np.asarray(fresh_problem.action_fn(jnp.asarray(clean_memory)))
        fresh_degraded = np.asarray(fresh_problem.action_fn(jnp.asarray(degraded_memory)))
        fresh_edited = np.asarray(fresh_problem.action_fn(jnp.asarray(edited_memory)))
        fresh.append(_recovery(fresh_clean, fresh_degraded, fresh_edited))

    report = {
        "experiment": "E12_reward_free_memory_recovery",
        "checkpoint": str(args.checkpoint),
        "data": str(args.data),
        "index": args.index,
        "preset": preset.name,
        "basis": args.basis,
        "basis_directions": int(basis_np.shape[0]),
        "response_rank": rank,
        "effective_rank_90": energy_rank(pullback.singular_values, threshold=0.9),
        "drop_fraction": args.drop_fraction,
        "dropped_history_rows": dropped,
        "trust_radius": args.trust_radius,
        "applied_relative_delta_norm": float(np.linalg.norm(delta) / max(np.linalg.norm(degraded_memory), 1e-12)),
        "linear_target_residual_before": float(np.linalg.norm(target)),
        "linear_target_residual_after": pullback.residual_norm,
        "same_noise_recovery": _recovery(clean_actions, degraded_actions, edited_actions),
        "negative_direction_recovery": _recovery(clean_actions, degraded_actions, negative_actions),
        "random_direction_recovery": _recovery(clean_actions, degraded_actions, random_actions),
        "fresh_noise_recoveries": fresh,
        "fresh_noise_mean_recovery": float(np.mean(fresh)) if fresh else None,
        "claim_scope": "reward-free recovery of clean-memory action behavior; not environment success",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    artifact_path = args.artifacts or args.output.with_suffix(".npz")
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        artifact_path,
        clean_memory=clean_memory,
        degraded_memory=degraded_memory,
        edited_memory=edited_memory,
        delta=delta,
        basis=basis_np,
        responses=responses_np,
        clean_actions=clean_actions,
        degraded_actions=degraded_actions,
        edited_actions=edited_actions,
        negative_actions=negative_actions,
        random_actions=random_actions,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
