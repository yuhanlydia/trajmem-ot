from __future__ import annotations

from typing import Any

import numpy as np


def vector_cosine(left: Any, right: Any, *, floor: float = 1e-12) -> float:
    """Return a finite cosine after flattening two action responses."""
    left_array = np.asarray(left, dtype=np.float64).reshape(-1)
    right_array = np.asarray(right, dtype=np.float64).reshape(-1)
    if left_array.shape != right_array.shape:
        raise ValueError(f"shape mismatch: {left_array.shape} != {right_array.shape}")
    if not np.isfinite(left_array).all() or not np.isfinite(right_array).all():
        raise ValueError("cosine inputs must be finite")
    denominator = max(np.linalg.norm(left_array) * np.linalg.norm(right_array), floor)
    return float(left_array @ right_array / denominator)


def action_slice_metrics(
    jvp: Any,
    chord: Any,
    *,
    robot_action_dim: int = 8,
) -> dict[str, float]:
    """Compare robot, padded, and full action channels separately."""
    jvp_array = np.asarray(jvp, dtype=np.float64)
    chord_array = np.asarray(chord, dtype=np.float64)
    if jvp_array.shape != chord_array.shape:
        raise ValueError(f"shape mismatch: {jvp_array.shape} != {chord_array.shape}")
    if jvp_array.ndim < 1 or robot_action_dim <= 0 or robot_action_dim > jvp_array.shape[-1]:
        raise ValueError("robot_action_dim must select a non-empty suffix of the action dimension")
    robot_jvp = jvp_array[..., :robot_action_dim]
    robot_chord = chord_array[..., :robot_action_dim]
    padded_jvp = jvp_array[..., robot_action_dim:]
    padded_chord = chord_array[..., robot_action_dim:]
    return {
        "cosine_robot_8d": vector_cosine(robot_jvp, robot_chord),
        "cosine_padded_24d": vector_cosine(padded_jvp, padded_chord),
        "cosine_all_32d": vector_cosine(jvp_array, chord_array),
        "jvp_norm_robot_8d": float(np.linalg.norm(robot_jvp)),
        "jvp_norm_padded_24d": float(np.linalg.norm(padded_jvp)),
        "jvp_norm_all_32d": float(np.linalg.norm(jvp_array)),
        "chord_norm_robot_8d": float(np.linalg.norm(robot_chord)),
        "chord_norm_padded_24d": float(np.linalg.norm(padded_chord)),
        "chord_norm_all_32d": float(np.linalg.norm(chord_array)),
    }


def primal_delta_norm(plain: Any, traced: Any) -> float:
    """Measure the FP32 difference between plain and JVP-traced primals."""
    plain_array = np.asarray(plain, dtype=np.float64)
    traced_array = np.asarray(traced, dtype=np.float64)
    if plain_array.shape != traced_array.shape:
        raise ValueError(f"shape mismatch: {plain_array.shape} != {traced_array.shape}")
    return float(np.linalg.norm(plain_array - traced_array))
