from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .jax_operator import quantized_action_chord, svd_ridge_pullback
from .memory_basis import combine_basis, normalize_basis

ArrayLike = Any
ActionFn = Callable[[ArrayLike], ArrayLike]


@dataclass(frozen=True)
class SecantColumnDiagnostics:
    direction_index: int
    nominal_relative_radius: float
    actual_relative_radius: float
    changed_fraction_plus: float
    changed_fraction_minus: float
    asymmetry: float
    input_half_norm: float
    action_half_norm: float


@dataclass(frozen=True)
class FiniteResponseModel:
    unit_directions: np.ndarray
    response_matrix: np.ndarray
    base_action: np.ndarray
    output_shape: tuple[int, ...]
    memory_norm: float
    diagnostics: tuple[SecantColumnDiagnostics, ...]


@dataclass(frozen=True)
class FinitePullbackResult:
    coefficients: np.ndarray
    nominal_delta: np.ndarray
    quantized_memory: np.ndarray
    applied_delta: np.ndarray
    applied_relative_norm: float
    linear_prediction: np.ndarray
    linear_residual_norm: float
    singular_values: np.ndarray
    rank: int


@dataclass(frozen=True)
class QuantizedMatchedMemory:
    """One finite-precision control matched in *applied* memory norm."""

    quantized_memory: np.ndarray
    applied_delta: np.ndarray
    applied_norm: float
    target_norm: float
    relative_error: float
    scale: float
    iterations: int
    changed_fraction: float


def _jax_modules():
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "JAX is required for deployed finite-response operators; install trajmem-ot[jax]"
        ) from exc
    return jax, jnp


def _block(tree: Any) -> Any:
    jax, _ = _jax_modules()
    return jax.tree.map(lambda value: value.block_until_ready(), tree)


def _slice_robot_actions(actions: np.ndarray, robot_action_dim: int) -> np.ndarray:
    if actions.ndim < 1:
        raise ValueError("actions must have at least one dimension")
    if robot_action_dim <= 0 or robot_action_dim > actions.shape[-1]:
        raise ValueError(
            f"robot_action_dim must be in [1, {actions.shape[-1]}], got {robot_action_dim}"
        )
    return actions[..., :robot_action_dim]


def quantized_norm_matched_memory(
    memory: ArrayLike,
    direction: np.ndarray,
    *,
    target_norm: float,
    relative_tolerance: float = 0.02,
    max_iterations: int = 48,
    max_scale_multiplier: float = 128.0,
) -> QuantizedMatchedMemory:
    """Match a control to a target norm after casting to the deployed dtype.

    Matching a dense FP32 delta before a BF16 cast is not a valid norm-matched
    control: different directions cross different quantization thresholds. This
    routine searches the scalar applied to a unit direction and retains the
    quantized endpoint whose *actual* delta norm is closest to ``target_norm``.
    """

    if not np.isfinite(target_norm) or target_norm <= 0:
        raise ValueError("target_norm must be a positive finite scalar")
    if not np.isfinite(relative_tolerance) or not 0 < relative_tolerance < 1:
        raise ValueError("relative_tolerance must lie in (0, 1)")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if not np.isfinite(max_scale_multiplier) or max_scale_multiplier <= 1:
        raise ValueError("max_scale_multiplier must exceed one")

    _, jnp = _jax_modules()
    memory_jax = jnp.asarray(memory)
    memory_fp32 = np.asarray(memory_jax, dtype=np.float32)
    direction_fp32 = np.asarray(direction, dtype=np.float32)
    if direction_fp32.shape != memory_fp32.shape:
        raise ValueError(
            f"direction shape {direction_fp32.shape} != memory shape {memory_fp32.shape}"
        )
    if not np.isfinite(direction_fp32).all():
        raise ValueError("direction contains non-finite values")
    direction_norm = float(np.linalg.norm(direction_fp32))
    if direction_norm <= 1e-12:
        raise ValueError("direction has zero-norm")
    unit = direction_fp32 / direction_norm

    def evaluate(scale: float) -> tuple[float, np.ndarray, np.ndarray]:
        quantized = jnp.asarray(memory_fp32 + float(scale) * unit, dtype=memory_jax.dtype)
        _block(quantized)
        quantized_np = np.asarray(quantized, dtype=np.float32)
        applied = quantized_np - memory_fp32
        return float(np.linalg.norm(applied)), quantized_np, applied

    evaluations = 0
    low = 0.0
    high = float(target_norm)
    best: tuple[float, float, np.ndarray, np.ndarray] | None = None

    def consider(scale: float) -> float:
        nonlocal best, evaluations
        norm, quantized, applied = evaluate(scale)
        evaluations += 1
        error = abs(norm - target_norm)
        if best is None or error < best[0]:
            best = (error, scale, quantized, applied)
        return norm

    high_norm = consider(high)
    maximum_scale = float(target_norm) * float(max_scale_multiplier)
    while high_norm < target_norm and high < maximum_scale:
        low = high
        high = min(maximum_scale, high * 2.0)
        high_norm = consider(high)
        if high == maximum_scale:
            break

    if high_norm >= target_norm:
        for _ in range(max_iterations):
            midpoint = 0.5 * (low + high)
            midpoint_norm = consider(midpoint)
            if best is not None and best[0] / target_norm <= relative_tolerance:
                break
            if midpoint_norm < target_norm:
                low = midpoint
            else:
                high = midpoint

    assert best is not None
    error, scale, quantized_np, applied = best
    applied_norm = float(np.linalg.norm(applied))
    return QuantizedMatchedMemory(
        quantized_memory=quantized_np,
        applied_delta=applied,
        applied_norm=applied_norm,
        target_norm=float(target_norm),
        relative_error=float(error / target_norm),
        scale=float(scale),
        iterations=evaluations,
        changed_fraction=float(np.mean(quantized_np != memory_fp32)),
    )


def build_finite_response_model(
    action_fn: ActionFn,
    memory: ArrayLike,
    directions: np.ndarray,
    *,
    relative_radius: float,
    robot_action_dim: int = 8,
    minimum_actual_norm: float = 1e-12,
) -> FiniteResponseModel:
    """Build a response matrix from actual finite-precision endpoint chords."""

    if not np.isfinite(relative_radius) or relative_radius <= 0:
        raise ValueError("relative_radius must be a positive finite scalar")
    if not np.isfinite(minimum_actual_norm) or minimum_actual_norm <= 0:
        raise ValueError("minimum_actual_norm must be positive and finite")

    _, jnp = _jax_modules()
    memory_jax = jnp.asarray(memory)
    memory_np = np.asarray(memory_jax, dtype=np.float32)
    memory_norm = float(np.linalg.norm(memory_np))
    if memory_norm <= minimum_actual_norm:
        raise ValueError("memory must have nonzero Frobenius norm")

    basis = normalize_basis(np.asarray(directions, dtype=np.float32))
    if basis.shape[1:] != memory_np.shape:
        raise ValueError(
            f"direction shape {basis.shape[1:]} != memory shape {memory_np.shape}"
        )

    base = action_fn(memory_jax)
    _block(base)
    base_np = _slice_robot_actions(np.asarray(base, dtype=np.float32), robot_action_dim)

    unit_directions: list[np.ndarray] = []
    response_columns: list[np.ndarray] = []
    diagnostics: list[SecantColumnDiagnostics] = []
    step = relative_radius * memory_norm

    for index, direction in enumerate(basis):
        action_plus, action_minus, actual_half, memory_plus, memory_minus = (
            quantized_action_chord(
                action_fn,
                memory_jax,
                jnp.asarray(direction, dtype=memory_jax.dtype),
                step=step,
            )
        )
        _block((action_plus, action_minus, actual_half, memory_plus, memory_minus))

        plus_memory = np.asarray(memory_plus, dtype=np.float32)
        minus_memory = np.asarray(memory_minus, dtype=np.float32)
        actual_half_np = np.asarray(actual_half, dtype=np.float32)
        actual_norm = float(np.linalg.norm(actual_half_np))
        if actual_norm <= minimum_actual_norm:
            raise ValueError(
                f"direction {index} is erased by the deployed memory dtype at radius {relative_radius}"
            )

        plus_action = _slice_robot_actions(
            np.asarray(action_plus, dtype=np.float32), robot_action_dim
        )
        minus_action = _slice_robot_actions(
            np.asarray(action_minus, dtype=np.float32), robot_action_dim
        )
        action_half = (plus_action - minus_action) / 2.0

        changed_plus = float(np.mean(plus_memory != memory_np))
        changed_minus = float(np.mean(minus_memory != memory_np))
        denominator = max(float(np.linalg.norm(plus_memory - minus_memory)), 1e-12)
        asymmetry = float(
            np.linalg.norm((plus_memory - memory_np) + (minus_memory - memory_np))
            / denominator
        )

        unit_directions.append(actual_half_np / actual_norm)
        response_columns.append(action_half.reshape(-1) / actual_norm)
        diagnostics.append(
            SecantColumnDiagnostics(
                direction_index=index,
                nominal_relative_radius=float(relative_radius),
                actual_relative_radius=actual_norm / memory_norm,
                changed_fraction_plus=changed_plus,
                changed_fraction_minus=changed_minus,
                asymmetry=asymmetry,
                input_half_norm=actual_norm,
                action_half_norm=float(np.linalg.norm(action_half)),
            )
        )

    unit_array = normalize_basis(np.stack(unit_directions).astype(np.float32))
    response_matrix = np.stack(response_columns, axis=1).astype(np.float64)
    return FiniteResponseModel(
        unit_directions=unit_array,
        response_matrix=response_matrix,
        base_action=base_np,
        output_shape=tuple(int(value) for value in base_np.shape),
        memory_norm=memory_norm,
        diagnostics=tuple(diagnostics),
    )


def solve_finite_response_pullback(
    model: FiniteResponseModel,
    memory: ArrayLike,
    target_delta: np.ndarray,
    *,
    damping: float,
    rank: int | None,
    max_relative_norm: float,
) -> FinitePullbackResult:
    """Solve and quantize a bounded update in a finite-response subspace."""

    if not np.isfinite(max_relative_norm) or max_relative_norm <= 0:
        raise ValueError("max_relative_norm must be a positive finite scalar")
    target = np.asarray(target_delta, dtype=np.float64)
    if target.shape != model.output_shape:
        raise ValueError(
            f"target shape {target.shape} != response output shape {model.output_shape}"
        )

    pullback = svd_ridge_pullback(
        model.response_matrix,
        target.reshape(-1),
        damping=damping,
        rank=rank,
    )
    nominal_delta = combine_basis(model.unit_directions, pullback.coefficients).astype(
        np.float32
    )
    maximum = max_relative_norm * model.memory_norm
    nominal_norm = float(np.linalg.norm(nominal_delta))
    coefficient_scale = 1.0
    if nominal_norm > maximum:
        coefficient_scale = maximum / nominal_norm
        nominal_delta *= coefficient_scale

    _, jnp = _jax_modules()
    memory_jax = jnp.asarray(memory)
    memory_fp32 = np.asarray(memory_jax, dtype=np.float32)
    quantized = jnp.asarray(memory_fp32 + nominal_delta, dtype=memory_jax.dtype)
    _block(quantized)
    quantized_np = np.asarray(quantized, dtype=np.float32)
    applied_delta = quantized_np - memory_fp32
    applied_relative_norm = float(
        np.linalg.norm(applied_delta) / max(model.memory_norm, 1e-12)
    )

    clipped_coefficients = pullback.coefficients * coefficient_scale
    linear_prediction = model.response_matrix @ clipped_coefficients
    linear_residual = float(np.linalg.norm(linear_prediction - target.reshape(-1)))

    return FinitePullbackResult(
        coefficients=np.asarray(clipped_coefficients, dtype=np.float64),
        nominal_delta=nominal_delta,
        quantized_memory=quantized_np,
        applied_delta=applied_delta,
        applied_relative_norm=applied_relative_norm,
        linear_prediction=linear_prediction,
        linear_residual_norm=linear_residual,
        singular_values=pullback.singular_values,
        rank=pullback.rank,
    )
