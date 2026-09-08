from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

ArrayLike = Any
ActionFn = Callable[[ArrayLike], ArrayLike]


@dataclass(frozen=True)
class PullbackResult:
    """Solution of a response-subspace ridge inverse problem."""

    coefficients: np.ndarray
    prediction: np.ndarray
    residual_norm: float
    singular_values: np.ndarray
    rank: int


def _jax_modules():
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:  # pragma: no cover - exercised on minimal installs
        raise RuntimeError(
            "JAX is required for action-response operators; install trajmem-ot[jax]"
        ) from exc
    return jax, jnp


def _shape_tuple(value: ArrayLike) -> tuple[int, ...]:
    return tuple(int(dim) for dim in np.shape(value))


def action_jvp(
    action_fn: ActionFn, memory: ArrayLike, direction: ArrayLike
) -> tuple[ArrayLike, ArrayLike]:
    """Return ``F(memory)`` and the exact forward-mode response ``J_M direction``.

    ``action_fn`` should close over the frozen policy parameters, current
    observation, instruction, robot state, and a specified diffusion-noise
    tensor. Only ``memory`` is treated as a differentiable argument.
    """
    if _shape_tuple(direction) != _shape_tuple(memory):
        raise ValueError(
            f"direction shape {_shape_tuple(direction)} != memory shape {_shape_tuple(memory)}"
        )
    jax, jnp = _jax_modules()
    memory_array = jnp.asarray(memory)
    # Checkpoint memories are commonly BF16 while basis construction uses FP32.
    # JAX forward-mode AD requires primal and tangent dtypes to match exactly.
    direction_array = jnp.asarray(direction, dtype=memory_array.dtype)
    return jax.jvp(action_fn, (memory_array,), (direction_array,))


def batched_action_jvps(
    action_fn: ActionFn,
    memory: ArrayLike,
    directions: ArrayLike,
    *,
    chunk_size: int,
) -> tuple[ArrayLike, ArrayLike]:
    """Compute exact JVP responses in memory-safe direction chunks.

    The direction dimension is the preferred batch dimension on 16GB/24GB
    hardware because all directions share the same model parameters and current
    inputs. The final short chunk is compiled separately when necessary.
    """
    jax, jnp = _jax_modules()
    memory_array = jnp.asarray(memory)
    directions_array = jnp.asarray(directions, dtype=memory_array.dtype)
    if directions_array.ndim < 2:
        raise ValueError("directions must have shape [K, ...memory_shape]")
    if _shape_tuple(directions_array)[1:] != _shape_tuple(memory):
        raise ValueError(
            f"direction shape {_shape_tuple(directions_array)[1:]} != memory shape {_shape_tuple(memory)}"
        )
    if directions_array.shape[0] == 0:
        raise ValueError("at least one direction is required")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    base_actions = action_fn(memory_array)

    def one(direction):
        return jax.jvp(action_fn, (memory_array,), (direction,))[1]

    responses = []
    for start in range(0, int(directions_array.shape[0]), chunk_size):
        chunk = directions_array[start : start + chunk_size]
        responses.append(jax.vmap(one)(chunk))
    return base_actions, jnp.concatenate(responses, axis=0)


def central_action_secant(
    action_fn: ActionFn,
    memory: ArrayLike,
    direction: ArrayLike,
    *,
    step: float,
) -> ArrayLike:
    """Central finite-difference action response used only to validate JVPs."""
    if _shape_tuple(direction) != _shape_tuple(memory):
        raise ValueError(
            f"direction shape {_shape_tuple(direction)} != memory shape {_shape_tuple(memory)}"
        )
    if not np.isfinite(step) or step <= 0:
        raise ValueError("step must be a positive finite scalar")
    _, jnp = _jax_modules()
    memory_array = jnp.asarray(memory)
    direction_array = jnp.asarray(direction, dtype=memory_array.dtype)
    return (
        action_fn(memory_array + step * direction_array)
        - action_fn(memory_array - step * direction_array)
    ) / (2.0 * step)


def energy_rank(singular_values: np.ndarray, *, threshold: float = 0.9) -> int:
    """Smallest rank explaining ``threshold`` of squared singular-value energy."""
    values = np.asarray(singular_values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("singular_values must be one-dimensional")
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")
    if values.size == 0:
        return 0
    energy = np.square(values)
    total = float(energy.sum())
    if total <= 0.0:
        return 0
    cumulative = np.cumsum(energy) / total
    return int(np.searchsorted(cumulative, threshold, side="left") + 1)


def response_svd(response_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SVD of ``Y = J_M D`` where columns correspond to memory directions."""
    matrix = np.asarray(response_matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("response_matrix must have shape [output_dim, basis_dim]")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("response_matrix cannot be empty")
    if not np.isfinite(matrix).all():
        raise ValueError("response_matrix contains non-finite values")
    return np.linalg.svd(matrix, full_matrices=False)


def svd_ridge_pullback(
    response_matrix: np.ndarray,
    target: np.ndarray,
    *,
    damping: float,
    rank: int | None = None,
) -> PullbackResult:
    """Solve ``min_c ||Yc-u||^2 + damping ||c||^2`` via truncated SVD.

    ``response_matrix`` is ``Y = J_M D``. The returned coefficients parameterize
    a memory edit in the supplied basis: ``Delta M = D c``.
    """
    matrix = np.asarray(response_matrix, dtype=np.float64)
    target_vector = np.asarray(target, dtype=np.float64).reshape(-1)
    if matrix.ndim != 2:
        raise ValueError("response_matrix must have shape [output_dim, basis_dim]")
    if matrix.shape[0] != target_vector.size:
        raise ValueError(
            f"target has {target_vector.size} entries but response output has {matrix.shape[0]}"
        )
    if damping < 0 or not np.isfinite(damping):
        raise ValueError("damping must be a non-negative finite scalar")
    if not np.isfinite(matrix).all() or not np.isfinite(target_vector).all():
        raise ValueError("response_matrix and target must be finite")

    u, singular_values, vh = response_svd(matrix)
    max_rank = singular_values.size
    used_rank = max_rank if rank is None else int(rank)
    if used_rank <= 0 or used_rank > max_rank:
        raise ValueError(f"rank must be in [1, {max_rank}]")

    u_r = u[:, :used_rank]
    s_r = singular_values[:used_rank]
    vh_r = vh[:used_rank]
    filter_factors = s_r / (np.square(s_r) + damping)
    coefficients = vh_r.T @ (filter_factors * (u_r.T @ target_vector))
    prediction = matrix @ coefficients
    residual_norm = float(np.linalg.norm(prediction - target_vector))
    return PullbackResult(
        coefficients=coefficients,
        prediction=prediction,
        residual_norm=residual_norm,
        singular_values=singular_values,
        rank=used_rank,
    )
