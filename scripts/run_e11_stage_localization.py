#!/usr/bin/env python3
"""E11-C stage localization for the released modulation checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from trajmem_ot.action_diagnostics import action_slice_metrics, primal_delta_norm
from trajmem_ot.jax_operator import action_jvp, quantized_action_chord
from trajmem_ot.robomme_jax import replace_observation_memory
from trajmem_ot.robomme_runtime import load_runtime_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-steps", type=int, default=1)
    parser.add_argument("--relative-radius", type=float, default=2.5e-4)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _block(tree):
    import jax

    return jax.tree.map(lambda value: value.block_until_ready(), tree)


def _cosine(left, right) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    return float(left @ right / max(np.linalg.norm(left) * np.linalg.norm(right), 1e-12))


def main() -> None:
    args = parse_args()
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import einops
    import jax
    import jax.numpy as jnp
    from mme_vla_suite.models.integration.history_observation import preprocess_observation
    from mme_vla_suite.models.integration.history_pi0 import make_attn_mask

    runtime = load_runtime_state(
        checkpoint=args.checkpoint,
        data=args.data,
        index=args.index,
        seed=args.seed,
        train_config_name="mme_vla_suite",
    )
    model = runtime.policy._model
    processed_observation = preprocess_observation(None, runtime.observation, train=False)
    memory = processed_observation.static_image_emb[0]
    memory_norm = float(np.linalg.norm(np.asarray(memory, dtype=np.float32)))
    noise = jax.random.normal(
        jax.random.key(args.seed + 501),
        (1, int(model.action_horizon), int(model.action_dim)),
        dtype=jnp.float32,
    )
    x_t = noise
    time = jnp.asarray(1.0, dtype=jnp.float32)

    def memory_tokens_fn(unbatched_memory):
        edited = replace_observation_memory(
            processed_observation,
            unbatched_memory[None, ...],
            field="static_image_emb",
        )
        tokens, _, _, _, _ = model.embed_memory(edited)
        return tokens[0]

    base_tokens = memory_tokens_fn(memory)
    zero_memory = jnp.zeros_like(memory)
    traced_tokens, tokens_jvp_zero = action_jvp(memory_tokens_fn, memory, zero_memory)
    _block((base_tokens, traced_tokens, tokens_jvp_zero))

    prefix_tokens, prefix_mask, prefix_ar_mask, _, _ = model.embed_prefix(processed_observation)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = model.PaliGemma.llm(
        [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
    )
    mem_seq, mem_mask, _, _, _ = model.embed_memory(processed_observation)
    suffix_tokens, suffix_mask, suffix_ar_mask, _, adarms_cond = model.embed_suffix(
        processed_observation, x_t, jnp.broadcast_to(time, (1,))
    )
    suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
    prefix_for_suffix = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
    full_attn_mask = jnp.concatenate([prefix_for_suffix, suffix_attn_mask], axis=-1)
    positions_suffix = (
        jnp.sum(prefix_mask, axis=-1)[:, None]
        + jnp.cumsum(suffix_mask, axis=-1)
        - 1
    )

    def velocity_fn(tokens):
        (prefix_out, suffix_out), _ = model.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions_suffix,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
            mem_seq=[None, tokens[None, ...]],
            mem_mask=[None, mem_mask],
        )
        del prefix_out
        return model.action_out_proj(suffix_out[:, -model.action_horizon :])[0]

    base_velocity = velocity_fn(base_tokens)
    traced_velocity, velocity_jvp_zero = action_jvp(
        velocity_fn, base_tokens, jnp.zeros_like(base_tokens)
    )
    _block((base_velocity, traced_velocity, velocity_jvp_zero))

    direction = jax.random.normal(jax.random.key(args.seed + 777), memory.shape, dtype=memory.dtype)
    direction = direction / jnp.linalg.norm(direction)
    step = args.relative_radius * memory_norm
    action_plus, action_minus, actual_memory_half, memory_plus, memory_minus = quantized_action_chord(
        memory_tokens_fn, memory, direction, step=step
    )
    tokens_plus = memory_tokens_fn(memory_plus)
    tokens_minus = memory_tokens_fn(memory_minus)
    actual_tokens_half = (tokens_plus - tokens_minus) / jnp.asarray(2, dtype=tokens_plus.dtype)
    midpoint_tokens = (tokens_plus + tokens_minus) / jnp.asarray(2, dtype=tokens_plus.dtype)
    # Re-evaluate the same stage at the actual endpoint midpoint.
    velocity_plus = velocity_fn(tokens_plus)
    velocity_minus = velocity_fn(tokens_minus)
    velocity_chord = (velocity_plus - velocity_minus) / 2.0
    _, velocity_jvp_origin = action_jvp(velocity_fn, base_tokens, actual_tokens_half)
    _, velocity_jvp_midpoint = action_jvp(velocity_fn, midpoint_tokens, actual_tokens_half)
    _, token_jvp_origin = action_jvp(memory_tokens_fn, memory, actual_memory_half)
    _, token_jvp_midpoint = action_jvp(memory_tokens_fn, (memory_plus + memory_minus) / 2, actual_memory_half)
    _block(
        (
            action_plus,
            action_minus,
            actual_memory_half,
            token_jvp_origin,
            token_jvp_midpoint,
            velocity_plus,
            velocity_minus,
            velocity_jvp_origin,
            velocity_jvp_midpoint,
        )
    )
    token_chord = (tokens_plus - tokens_minus) / 2
    report = {
        "experiment": "E11-C_stage_localization",
        "checkpoint": str(args.checkpoint),
        "data": str(args.data),
        "index": args.index,
        "num_steps": args.num_steps,
        "relative_radius": args.relative_radius,
        "memory_shape": list(np.shape(memory)),
        "memory_dtype": str(memory.dtype),
        "tokens_shape": list(np.shape(base_tokens)),
        "tokens_dtype": str(base_tokens.dtype),
        "velocity_shape": list(np.shape(base_velocity)),
        "velocity_dtype": str(base_velocity.dtype),
        "memory_encoder_primal_delta_norm": primal_delta_norm(base_tokens, traced_tokens),
        "modulation_velocity_primal_delta_norm": primal_delta_norm(base_velocity, traced_velocity),
        "memory_encoder": {
            "jvp_origin_cosine": _cosine(np.asarray(token_jvp_origin), np.asarray(token_chord)),
            "jvp_midpoint_cosine": _cosine(np.asarray(token_jvp_midpoint), np.asarray(token_chord)),
            "jvp_norm": float(np.linalg.norm(np.asarray(token_jvp_origin))),
            "chord_norm": float(np.linalg.norm(np.asarray(token_chord))),
        },
        "memory_modulation_velocity": {
            "jvp_origin_cosine": _cosine(np.asarray(velocity_jvp_origin), np.asarray(velocity_chord)),
            "jvp_midpoint_cosine": _cosine(np.asarray(velocity_jvp_midpoint), np.asarray(velocity_chord)),
            "jvp_norm": float(np.linalg.norm(np.asarray(velocity_jvp_origin))),
            "chord_norm": float(np.linalg.norm(np.asarray(velocity_chord))),
            "robot_8d": action_slice_metrics(velocity_jvp_origin, velocity_chord),
        },
        "endpoint_changed_fraction_plus": float(np.mean(np.asarray(memory_plus) != np.asarray(memory))),
        "endpoint_changed_fraction_minus": float(np.mean(np.asarray(memory_minus) != np.asarray(memory))),
        "claim_scope": "stage localization only; no JVP correctness or environment claim",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
