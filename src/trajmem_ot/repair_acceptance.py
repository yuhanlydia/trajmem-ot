"""Per-edit held-out-noise rollback, separate from research continuation decisions."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class RepairAcceptance:
    accepted: bool
    median_improvement: float
    improving_fraction: float


def assess_repair(before_losses, after_losses, *, minimum_improving_fraction=.5):
    before = np.asarray(before_losses, dtype=np.float64)
    after = np.asarray(after_losses, dtype=np.float64)
    if (before.ndim != 1 or not before.size or before.shape != after.shape
            or not np.isfinite(before).all() or not np.isfinite(after).all()
            or np.any(before < 0) or np.any(after < 0)):
        raise ValueError("losses must be matching nonempty finite nonnegative vectors")
    if not 0 <= minimum_improving_fraction <= 1:
        raise ValueError("minimum_improving_fraction must lie in [0, 1]")
    improvement = before - after
    median = float(np.median(improvement))
    fraction = float(np.mean(improvement > 0))
    return RepairAcceptance(median > 0 and fraction >= minimum_improving_fraction,
                            median, fraction)
