from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .memory_basis import combine_basis

Array = np.ndarray


@dataclass(frozen=True)
class MemoryViewBatch:
    views: Array
    deltas: Array
    raw_delta_norms: Array
    applied_delta_norms: Array
    max_delta_norm: float


@dataclass(frozen=True)
class VarianceDecomposition:
    """Legacy one-way decomposition retained for backward compatibility."""

    total: float
    within_noise: float
    between_memory: float
    memory_fraction: float


@dataclass(frozen=True)
class MatchedVarianceDecomposition:
    """Balanced two-way decomposition for matched memory-view/noise grids.

    For actions ``A[b,n]`` evaluated with the same noise seeds for every memory
    view, the centered response is decomposed orthogonally into a memory-view
    main effect, a diffusion-noise main effect, and their interaction.
    """

    total: float
    memory_main: float
    noise_main: float
    interaction: float
    memory_fraction: float
    noise_fraction: float
    interaction_fraction: float


@dataclass(frozen=True)
class TargetCoverage:
    covered: bool
    covered_views: int
    total_views: int
    covered_samples: int
    total_samples: int
    min_distance: float


def _validate_action_grid(actions: Array) -> Array:
    array = np.asarray(actions, dtype=np.float64)
    if array.ndim < 3:
        raise ValueError("actions must have shape [views, noises, ...action_shape]")
    if array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError("actions require at least one view and one noise sample")
    if not np.isfinite(array).all():
        raise ValueError("actions contain non-finite values")
    return array


def build_memory_views(
    memory: Array,
    basis: Array,
    coefficients: Array,
    *,
    relative_radius: float,
) -> MemoryViewBatch:
    """Construct memory views inside one Frobenius trust region."""

    memory_array = np.asarray(memory)
    basis_array = np.asarray(basis)
    coefficient_array = np.asarray(coefficients)
    if memory_array.ndim < 1:
        raise ValueError("memory must have at least one dimension")
    if basis_array.ndim != memory_array.ndim + 1 or basis_array.shape[1:] != memory_array.shape:
        raise ValueError(
            f"basis must have shape [K, {memory_array.shape}], got {basis_array.shape}"
        )
    if coefficient_array.ndim != 2 or coefficient_array.shape[1] != basis_array.shape[0]:
        raise ValueError(
            f"coefficients must have shape [views, {basis_array.shape[0]}], got {coefficient_array.shape}"
        )
    if not np.isfinite(relative_radius) or relative_radius < 0:
        raise ValueError("relative_radius must be a non-negative finite scalar")
    if not np.isfinite(memory_array).all() or not np.isfinite(coefficient_array).all():
        raise ValueError("memory and coefficients must be finite")

    raw_deltas = np.stack(
        [combine_basis(basis_array, row) for row in coefficient_array], axis=0
    )
    raw_norms = np.linalg.norm(raw_deltas.reshape(raw_deltas.shape[0], -1), axis=1)
    max_norm = float(relative_radius * np.linalg.norm(memory_array))
    scales = np.ones_like(raw_norms)
    nonzero = raw_norms > 0
    if max_norm == 0.0:
        scales[nonzero] = 0.0
    else:
        scales[nonzero] = np.minimum(1.0, max_norm / raw_norms[nonzero])
    reshape = (raw_deltas.shape[0],) + (1,) * memory_array.ndim
    deltas = raw_deltas * scales.reshape(reshape)
    views = memory_array[None, ...] + deltas
    applied_norms = np.linalg.norm(deltas.reshape(deltas.shape[0], -1), axis=1)
    return MemoryViewBatch(
        views=views,
        deltas=deltas,
        raw_delta_norms=raw_norms,
        applied_delta_norms=applied_norms,
        max_delta_norm=max_norm,
    )


def variance_decomposition(actions: Array) -> VarianceDecomposition:
    """Legacy law-of-total-variance summary.

    This mixes the diffusion main effect and memory-by-noise interaction in the
    within-view term. Use :func:`matched_variance_decomposition` for balanced,
    seed-matched branching experiments.
    """

    array = _validate_action_grid(actions)
    flat = array.reshape(array.shape[0], array.shape[1], -1)
    view_means = flat.mean(axis=1)
    overall_mean = flat.mean(axis=(0, 1))
    within = float(np.mean(np.sum(np.square(flat - view_means[:, None, :]), axis=-1)))
    between = float(np.mean(np.sum(np.square(view_means - overall_mean[None, :]), axis=-1)))
    total = float(np.mean(np.sum(np.square(flat - overall_mean[None, None, :]), axis=-1)))
    return VarianceDecomposition(
        total=total,
        within_noise=within,
        between_memory=between,
        memory_fraction=between / total if total > 0 else 0.0,
    )


def matched_variance_decomposition(actions: Array) -> MatchedVarianceDecomposition:
    """Orthogonally decompose a balanced ``[view, noise, ...]`` response grid.

    The same ordered noise seeds must be used for every view. Under that design,
    the decomposition is exact in sample space:

    ``total = memory_main + noise_main + interaction``.

    Unlike the legacy one-way fraction, this does not fold the shared noise main
    effect into a view-dependent term when allocations use different numbers of
    noise samples.
    """

    array = _validate_action_grid(actions)
    flat = array.reshape(array.shape[0], array.shape[1], -1)
    grand = flat.mean(axis=(0, 1))
    memory_effect = flat.mean(axis=1) - grand[None, :]
    noise_effect = flat.mean(axis=0) - grand[None, :]
    interaction = (
        flat
        - grand[None, None, :]
        - memory_effect[:, None, :]
        - noise_effect[None, :, :]
    )

    total = float(np.mean(np.sum(np.square(flat - grand[None, None, :]), axis=-1)))
    memory_main = float(np.mean(np.sum(np.square(memory_effect), axis=-1)))
    noise_main = float(np.mean(np.sum(np.square(noise_effect), axis=-1)))
    interaction_value = float(np.mean(np.sum(np.square(interaction), axis=-1)))

    memory_main = max(memory_main, 0.0)
    noise_main = max(noise_main, 0.0)
    interaction_value = max(interaction_value, 0.0)
    if total <= 0.0:
        memory_fraction = noise_fraction = interaction_fraction = 0.0
    else:
        memory_fraction = memory_main / total
        noise_fraction = noise_main / total
        interaction_fraction = interaction_value / total

    return MatchedVarianceDecomposition(
        total=total,
        memory_main=memory_main,
        noise_main=noise_main,
        interaction=interaction_value,
        memory_fraction=memory_fraction,
        noise_fraction=noise_fraction,
        interaction_fraction=interaction_fraction,
    )


def nearest_target_coverage(
    actions: Array, target: Array, *, tolerance: float
) -> TargetCoverage:
    """Measure whether any memory/noise branch reaches a target action mode."""

    array = _validate_action_grid(actions)
    target_array = np.asarray(target, dtype=np.float64)
    if array.shape[2:] != target_array.shape:
        raise ValueError(
            f"target shape {target_array.shape} != action shape {array.shape[2:]}"
        )
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be a non-negative finite scalar")
    distances = np.linalg.norm(
        (array - target_array[None, None, ...]).reshape(array.shape[0], array.shape[1], -1),
        axis=-1,
    )
    covered_mask = distances <= tolerance
    covered_views = int(np.sum(np.any(covered_mask, axis=1)))
    covered_samples = int(np.sum(covered_mask))
    return TargetCoverage(
        covered=bool(covered_samples > 0),
        covered_views=covered_views,
        total_views=int(array.shape[0]),
        covered_samples=covered_samples,
        total_samples=int(array.shape[0] * array.shape[1]),
        min_distance=float(distances.min()),
    )


def allocation_grid(
    total_budget: int, *, memory_view_counts: Sequence[int]
) -> tuple[tuple[int, int], ...]:
    """Return matched-compute ``(memory_views, noises_per_view)`` allocations."""

    if total_budget <= 0:
        raise ValueError("total_budget must be positive")
    allocations = []
    for views in memory_view_counts:
        value = int(views)
        if value <= 0:
            raise ValueError("memory view counts must be positive")
        if total_budget % value != 0:
            raise ValueError(f"memory view count {value} must divide total budget {total_budget}")
        allocations.append((value, total_budget // value))
    return tuple(allocations)


def hypothesis_coefficients(
    *, view_count: int, basis_size: int, seed: int, include_base: bool = True
) -> Array:
    """Generate symmetric unit coefficient vectors for competing memory views."""

    if view_count <= 0 or basis_size <= 0:
        raise ValueError("view_count and basis_size must be positive")
    remaining = view_count - int(include_base)
    if remaining < 0 or remaining % 2 != 0:
        raise ValueError("non-base memory views must form symmetric pairs")
    rng = np.random.default_rng(seed)
    rows: list[Array] = []
    if include_base:
        rows.append(np.zeros(basis_size, dtype=np.float64))
    for _ in range(remaining // 2):
        vector = rng.normal(size=basis_size)
        norm = np.linalg.norm(vector)
        if norm <= 1e-12:
            vector[0] = 1.0
            norm = 1.0
        vector = vector / norm
        rows.extend([vector, -vector])
    return np.stack(rows, axis=0)


def contiguous_history_mask_views(
    mask: Array,
    *,
    view_count: int,
    keep_fraction: float,
    include_full: bool = False,
) -> Array:
    """Create readout views from contiguous valid history spans."""

    base = np.asarray(mask, dtype=bool)
    if base.ndim != 1:
        raise ValueError("mask must be one-dimensional")
    if view_count <= 0:
        raise ValueError("view_count must be positive")
    if not 0.0 < keep_fraction <= 1.0:
        raise ValueError("keep_fraction must be in (0, 1]")
    valid = np.flatnonzero(base)
    if valid.size == 0:
        raise ValueError("mask contains no valid history tokens")
    window_count = view_count - int(include_full)
    if window_count < 0:
        raise ValueError("include_full requires at least one view")
    keep = max(1, min(valid.size, int(round(valid.size * keep_fraction))))
    rows: list[Array] = []
    if include_full:
        rows.append(base.copy())
    if window_count:
        max_start = valid.size - keep
        starts = np.rint(np.linspace(0, max_start, num=window_count)).astype(int)
        for start in starts:
            row = np.zeros_like(base)
            row[valid[start : start + keep]] = True
            rows.append(row)
    return np.stack(rows, axis=0)
