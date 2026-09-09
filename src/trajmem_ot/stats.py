from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float
    confidence: float
    resamples: int


def percentile_bootstrap(
    values: np.ndarray,
    *,
    statistic: Callable[[np.ndarray], float] = np.mean,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
) -> BootstrapInterval:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("values must be a non-empty one-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError("values contain non-finite entries")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0, 1)")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(resamples, array.size))
    estimates = np.asarray([statistic(array[index]) for index in indices], dtype=np.float64)
    alpha = (1.0 - confidence) / 2.0
    return BootstrapInterval(
        estimate=float(statistic(array)),
        lower=float(np.quantile(estimates, alpha)),
        upper=float(np.quantile(estimates, 1.0 - alpha)),
        confidence=float(confidence),
        resamples=int(resamples),
    )


def exact_two_sided_sign_test(values: np.ndarray, *, zero_tolerance: float = 0.0) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if not np.isfinite(array).all():
        raise ValueError("values contain non-finite entries")
    wins = int(np.sum(array > zero_tolerance))
    losses = int(np.sum(array < -zero_tolerance))
    ties = int(array.size - wins - losses)
    n = wins + losses
    if n == 0:
        p_value = 1.0
    else:
        tail = sum(comb(n, k) for k in range(0, min(wins, losses) + 1)) / (2**n)
        p_value = min(1.0, 2.0 * tail)
    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "p_value": float(p_value),
    }
