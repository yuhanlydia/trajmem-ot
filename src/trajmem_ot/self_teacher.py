"""Action targets from temporary history views of the same frozen policy."""
from __future__ import annotations

import numpy as np

from .transport import sinkhorn


def distillation_targets(student_actions, teacher_actions, *, mode="paired",
                         teacher_weights=None, epsilon=None):
    student = np.asarray(student_actions, dtype=np.float64)
    teacher = np.asarray(teacher_actions, dtype=np.float64)
    if (student.ndim != 3 or teacher.ndim != 3
            or student.shape[1:] != teacher.shape[1:]
            or not student.size or not teacher.size
            or not np.isfinite(student).all() or not np.isfinite(teacher).all()):
        raise ValueError("actions must be finite, nonempty [particles, horizon, action_dim]")
    weights = (np.ones(len(teacher)) if teacher_weights is None
               else np.asarray(teacher_weights, dtype=np.float64))
    if (weights.shape != (len(teacher),) or not np.isfinite(weights).all()
            or np.any(weights < 0) or weights.sum() <= 0):
        raise ValueError("teacher weights must be finite nonnegative masses with positive sum")
    weights = weights / weights.sum()
    if mode == "paired":
        if student.shape != teacher.shape:
            raise ValueError("paired targets require matching particles in identical noise order")
        if teacher_weights is not None:
            raise ValueError("teacher weights apply to centroid and OT targets only")
        return teacher - student
    if mode == "centroid":
        return np.tensordot(weights, teacher, axes=1) - student
    if mode != "ot":
        raise ValueError("target mode must be paired, centroid, or ot")
    active = weights > 0
    teacher_flat = teacher[active].reshape(active.sum(), -1)
    student_flat = student.reshape(len(student), -1)
    cost = np.mean((student_flat[:, None] - teacher_flat[None]) ** 2, axis=-1)
    positive = cost[cost > 0]
    used_epsilon = (float(epsilon) if epsilon is not None
                    else .05 * float(np.median(positive)) if positive.size else .05)
    source = np.full(len(student), 1. / len(student))
    coupling = sinkhorn(source, weights[active], cost, epsilon=used_epsilon, iterations=1000)
    # Normalize realized rows; finite Sinkhorn iterations need not reach exact marginals.
    mass = coupling.sum(axis=1, keepdims=True)
    if np.any(mass <= 0):
        raise ValueError("OT produced an empty source row; increase epsilon")
    return (coupling @ teacher_flat / mass - student_flat).reshape(student.shape)
