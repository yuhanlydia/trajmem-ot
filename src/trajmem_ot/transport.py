from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrajectoryTransportResult:
    source_weights: np.ndarray
    target_weights: np.ndarray
    cost: np.ndarray
    coupling: np.ndarray
    transport: np.ndarray
    epsilon: float


def trajectory_cost(actions: np.ndarray, *, velocity_weight: float = 0.25) -> np.ndarray:
    array = np.asarray(actions, dtype=np.float64)
    if array.ndim != 3 or array.shape[0] < 1:
        raise ValueError("actions must have shape [particles, horizon, action_dim]")
    if not np.isfinite(array).all():
        raise ValueError("actions contain non-finite values")
    flat = array.reshape(array.shape[0], -1)
    differences = flat[:, None, :] - flat[None, :, :]
    position = np.sum(np.square(differences), axis=-1) / array.shape[1]
    if array.shape[1] < 2:
        return position
    velocity = np.diff(array, axis=1).reshape(array.shape[0], -1)
    velocity_diff = velocity[:, None, :] - velocity[None, :, :]
    velocity_cost = np.sum(np.square(velocity_diff), axis=-1) / (array.shape[1] - 1)
    return position + float(velocity_weight) * velocity_cost


def return_tilted_weights(returns: np.ndarray, *, beta: float) -> np.ndarray:
    values = np.asarray(returns, dtype=np.float64)
    if values.ndim != 1 or values.size < 1:
        raise ValueError("returns must be a non-empty one-dimensional array")
    if not np.isfinite(values).all() or not np.isfinite(beta):
        raise ValueError("returns and beta must be finite")
    std = float(values.std())
    normalized = (values - values.mean()) / max(std, 1e-6)
    logits = float(beta) * normalized
    logits -= logits.max()
    weights = np.exp(logits)
    return weights / weights.sum()


def _logsumexp(values: np.ndarray, *, axis: int) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    reduced = maximum + np.log(np.sum(np.exp(values - maximum), axis=axis, keepdims=True))
    return np.squeeze(reduced, axis=axis)


def sinkhorn(
    source: np.ndarray,
    target: np.ndarray,
    cost: np.ndarray,
    *,
    epsilon: float,
    iterations: int = 100,
) -> np.ndarray:
    p = np.asarray(source, dtype=np.float64)
    q = np.asarray(target, dtype=np.float64)
    matrix = np.asarray(cost, dtype=np.float64)
    if p.ndim != 1 or q.ndim != 1 or matrix.shape != (p.size, q.size):
        raise ValueError("source, target, and cost shapes are inconsistent")
    if np.any(p <= 0) or np.any(q <= 0):
        raise ValueError("source and target weights must be strictly positive")
    if not np.isclose(p.sum(), 1.0) or not np.isclose(q.sum(), 1.0):
        raise ValueError("source and target weights must sum to one")
    if not np.isfinite(matrix).all() or not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("cost must be finite and epsilon positive")
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    kernel = -matrix / float(epsilon)
    log_p = np.log(p)
    log_q = np.log(q)
    log_u = np.zeros_like(p)
    log_v = np.zeros_like(q)
    for _ in range(iterations):
        log_u = log_p - _logsumexp(kernel + log_v[None, :], axis=1)
        log_v = log_q - _logsumexp(kernel + log_u[:, None], axis=0)
    coupling = np.exp(kernel + log_u[:, None] + log_v[None, :])
    return coupling


def return_tilted_transport(
    actions: np.ndarray,
    returns: np.ndarray,
    *,
    beta: float = 5.0,
    epsilon: float | None = None,
    sinkhorn_iterations: int = 100,
    velocity_weight: float = 0.25,
) -> TrajectoryTransportResult:
    array = np.asarray(actions, dtype=np.float64)
    values = np.asarray(returns, dtype=np.float64)
    if array.ndim != 3 or values.shape != (array.shape[0],):
        raise ValueError("actions must be [N,H,A] and returns must be [N]")
    source = np.full(array.shape[0], 1.0 / array.shape[0], dtype=np.float64)
    target = return_tilted_weights(values, beta=beta)
    cost = trajectory_cost(array, velocity_weight=velocity_weight)
    positive = cost[cost > 0]
    used_epsilon = (
        float(epsilon)
        if epsilon is not None
        else (0.05 * float(np.median(positive)) if positive.size else 0.05)
    )
    coupling = sinkhorn(
        source,
        target,
        cost,
        epsilon=used_epsilon,
        iterations=sinkhorn_iterations,
    )
    flat = array.reshape(array.shape[0], -1)
    barycenter = coupling @ flat / source[:, None]
    transport = (barycenter - flat).reshape(array.shape)
    return TrajectoryTransportResult(
        source_weights=source,
        target_weights=target,
        cost=cost,
        coupling=coupling,
        transport=transport,
        epsilon=used_epsilon,
    )
