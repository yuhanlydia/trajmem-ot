from __future__ import annotations

from collections.abc import Sequence

import numpy as np

Array = np.ndarray


def _validate_basis(basis: Array) -> Array:
    array = np.asarray(basis)
    if array.ndim < 2:
        raise ValueError("basis must have shape [directions, ...memory_shape]")
    if array.shape[0] == 0:
        raise ValueError("basis must contain at least one direction")
    if not np.issubdtype(array.dtype, np.floating):
        array = array.astype(np.float32)
    if not np.isfinite(array).all():
        raise ValueError("basis contains non-finite values")
    return array


def normalize_basis(basis: Array, *, epsilon: float = 1e-12) -> Array:
    """Normalize each memory direction independently in Frobenius norm."""
    array = _validate_basis(basis)
    flat = array.reshape(array.shape[0], -1)
    norms = np.linalg.norm(flat, axis=1)
    if np.any(norms <= epsilon):
        raise ValueError("basis contains a zero-norm direction")
    return (flat / norms[:, None]).reshape(array.shape).astype(array.dtype, copy=False)


def random_rank_one_basis(
    memory_shape: Sequence[int], *, count: int, seed: int, dtype: np.dtype = np.float32
) -> Array:
    """Create deterministic unit-Frobenius rank-one directions for a 2-D memory.

    RoboMME perceptual history has shape ``[tokens, features]``. Restricting the
    legacy black-box controls to rank-one directions preserves the old
    experiment while making the basis explicit and reusable by JVP operators.
    """
    shape = tuple(int(value) for value in memory_shape)
    if len(shape) != 2 or any(value <= 0 for value in shape):
        raise ValueError("memory_shape must contain two positive dimensions")
    if count <= 0:
        raise ValueError("count must be positive")
    rng = np.random.default_rng(seed)
    directions = []
    for _ in range(count):
        temporal = rng.normal(size=(shape[0], 1))
        feature = rng.normal(size=(1, shape[1]))
        directions.append(temporal @ feature)
    return normalize_basis(np.stack(directions).astype(dtype, copy=False))


def history_difference_basis(differences: Array, *, count: int) -> Array:
    """Top right-singular directions of matched history-memory differences.

    ``differences`` has shape ``[pairs, ...memory_shape]``. The returned basis
    has shape ``[count, ...memory_shape]`` and is orthonormal after flattening.
    These directions represent real changes induced by alternative histories,
    unlike arbitrary activation-space noise.
    """
    array = np.asarray(differences)
    if array.ndim < 3:
        raise ValueError("differences must have shape [pairs, ...memory_shape]")
    if array.shape[0] < 1:
        raise ValueError("differences must contain at least one pair")
    max_count = min(array.shape[0], int(np.prod(array.shape[1:])))
    if count <= 0 or count > max_count:
        raise ValueError(f"count must be in [1, {max_count}]")
    if not np.isfinite(array).all():
        raise ValueError("differences contain non-finite values")

    flat = array.reshape(array.shape[0], -1).astype(np.float64, copy=False)
    # Centering isolates directions that vary across paired histories rather
    # than the mean offset shared by every pair. For one pair, preserve the
    # direction because centering would erase all signal.
    centered = flat - flat.mean(axis=0, keepdims=True) if flat.shape[0] > 1 else flat
    if np.linalg.norm(centered) <= 1e-12:
        centered = flat
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    nonzero = int(np.sum(singular_values > 1e-12))
    if nonzero < count:
        raise ValueError(f"count={count} exceeds numerical history-difference rank {nonzero}")
    basis = vh[:count].reshape((count,) + array.shape[1:])
    return normalize_basis(basis.astype(array.dtype if np.issubdtype(array.dtype, np.floating) else np.float32))


def combine_basis(basis: Array, coefficients: Array) -> Array:
    """Form ``Delta M = sum_k coefficients[k] * basis[k]``."""
    array = _validate_basis(basis)
    coeffs = np.asarray(coefficients)
    if coeffs.ndim != 1 or coeffs.shape[0] != array.shape[0]:
        raise ValueError(
            f"coefficients must have shape ({array.shape[0]},), got {coeffs.shape}"
        )
    if not np.isfinite(coeffs).all():
        raise ValueError("coefficients contain non-finite values")
    return np.tensordot(coeffs, array, axes=(0, 0))
