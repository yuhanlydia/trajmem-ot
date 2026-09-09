from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SetCoverage:
    tolerance: float
    coverage_fraction: float
    covered_targets: int
    total_targets: int
    mean_nearest_distance: float
    median_nearest_distance: float
    maximum_nearest_distance: float


@dataclass(frozen=True)
class HypothesisCoverageComparison:
    tolerance: float
    calibration: SetCoverage
    noise_only: SetCoverage
    branched: SetCoverage
    correct_only: SetCoverage | None
    coverage_gain: float
    mean_distance_improvement: float


@dataclass(frozen=True)
class CoverageCurvePoint:
    tolerance_multiplier: float
    comparison: HypothesisCoverageComparison


def _flatten_samples(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 2 or array.shape[0] < 1:
        raise ValueError(f"{name} must have shape [samples, ...] with at least one sample")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array.reshape(array.shape[0], -1)


def nearest_set_distances(candidates: np.ndarray, targets: np.ndarray) -> np.ndarray:
    candidate_flat = _flatten_samples(candidates, name="candidates")
    target_flat = _flatten_samples(targets, name="targets")
    if candidate_flat.shape[1] != target_flat.shape[1]:
        raise ValueError(
            f"candidate dimension {candidate_flat.shape[1]} != target dimension {target_flat.shape[1]}"
        )
    squared = np.sum(
        np.square(target_flat[:, None, :] - candidate_flat[None, :, :]), axis=-1
    )
    return np.sqrt(np.maximum(squared.min(axis=1), 0.0))


def coverage_from_distances(distances: np.ndarray, *, tolerance: float) -> SetCoverage:
    values = np.asarray(distances, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("distances must be a non-empty one-dimensional array")
    if not np.isfinite(values).all():
        raise ValueError("distances contain non-finite values")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be a non-negative finite scalar")
    covered = values <= tolerance
    return SetCoverage(
        tolerance=float(tolerance),
        coverage_fraction=float(np.mean(covered)),
        covered_targets=int(np.sum(covered)),
        total_targets=int(values.size),
        mean_nearest_distance=float(np.mean(values)),
        median_nearest_distance=float(np.median(values)),
        maximum_nearest_distance=float(np.max(values)),
    )


def calibrated_set_coverage(
    candidates: np.ndarray, targets: np.ndarray, *, tolerance: float
) -> SetCoverage:
    return coverage_from_distances(
        nearest_set_distances(candidates, targets), tolerance=tolerance
    )


def calibrated_tolerance(
    correct_reference: np.ndarray,
    correct_targets: np.ndarray,
    *,
    tolerance_quantile: float = 0.95,
    tolerance_multiplier: float = 1.25,
    minimum_tolerance: float = 1e-6,
) -> float:
    if not 0.0 < tolerance_quantile <= 1.0:
        raise ValueError("tolerance_quantile must be in (0, 1]")
    if not np.isfinite(tolerance_multiplier) or tolerance_multiplier <= 0:
        raise ValueError("tolerance_multiplier must be positive and finite")
    if not np.isfinite(minimum_tolerance) or minimum_tolerance <= 0:
        raise ValueError("minimum_tolerance must be positive and finite")
    distances = nearest_set_distances(correct_reference, correct_targets)
    return max(
        minimum_tolerance,
        float(np.quantile(distances, tolerance_quantile)) * tolerance_multiplier,
    )


def compare_hypothesis_coverage(
    correct_reference: np.ndarray,
    correct_targets: np.ndarray,
    noise_only: np.ndarray,
    branched: np.ndarray,
    *,
    correct_only: np.ndarray | None = None,
    tolerance_quantile: float = 0.95,
    tolerance_multiplier: float = 1.25,
    minimum_tolerance: float = 1e-6,
) -> HypothesisCoverageComparison:
    tolerance = calibrated_tolerance(
        correct_reference,
        correct_targets,
        tolerance_quantile=tolerance_quantile,
        tolerance_multiplier=tolerance_multiplier,
        minimum_tolerance=minimum_tolerance,
    )
    calibration_distances = nearest_set_distances(correct_reference, correct_targets)
    calibration = coverage_from_distances(calibration_distances, tolerance=tolerance)
    noise_result = calibrated_set_coverage(noise_only, correct_targets, tolerance=tolerance)
    branched_result = calibrated_set_coverage(branched, correct_targets, tolerance=tolerance)
    correct_result = (
        None
        if correct_only is None
        else calibrated_set_coverage(correct_only, correct_targets, tolerance=tolerance)
    )
    return HypothesisCoverageComparison(
        tolerance=tolerance,
        calibration=calibration,
        noise_only=noise_result,
        branched=branched_result,
        correct_only=correct_result,
        coverage_gain=branched_result.coverage_fraction - noise_result.coverage_fraction,
        mean_distance_improvement=(
            noise_result.mean_nearest_distance - branched_result.mean_nearest_distance
        ),
    )


def compare_hypothesis_coverage_curve(
    correct_reference: np.ndarray,
    correct_targets: np.ndarray,
    noise_only: np.ndarray,
    branched: np.ndarray,
    *,
    correct_only: np.ndarray | None = None,
    tolerance_quantile: float = 0.95,
    tolerance_multipliers: tuple[float, ...] = (0.75, 1.0, 1.25, 1.5, 2.0),
    minimum_tolerance: float = 1e-6,
) -> tuple[CoverageCurvePoint, ...]:
    """Evaluate the oracle conclusion over a predeclared tolerance grid."""

    if not tolerance_multipliers:
        raise ValueError("tolerance_multipliers must not be empty")
    normalized = tuple(float(value) for value in tolerance_multipliers)
    if any(not np.isfinite(value) or value <= 0 for value in normalized):
        raise ValueError("tolerance multipliers must be positive and finite")
    if tuple(sorted(normalized)) != normalized or len(set(normalized)) != len(normalized):
        raise ValueError("tolerance multipliers must be strictly increasing")
    return tuple(
        CoverageCurvePoint(
            tolerance_multiplier=multiplier,
            comparison=compare_hypothesis_coverage(
                correct_reference,
                correct_targets,
                noise_only,
                branched,
                correct_only=correct_only,
                tolerance_quantile=tolerance_quantile,
                tolerance_multiplier=multiplier,
                minimum_tolerance=minimum_tolerance,
            ),
        )
        for multiplier in normalized
    )


def headroom_recovery(
    *, noise_coverage: float, branched_coverage: float, epsilon: float = 1e-12
) -> float | None:
    """Fraction of support missing under shared memory recovered by branching."""

    noise = float(noise_coverage)
    branched = float(branched_coverage)
    if not (0.0 <= noise <= 1.0 and 0.0 <= branched <= 1.0):
        raise ValueError("coverage values must lie in [0, 1]")
    headroom = 1.0 - noise
    if headroom <= epsilon:
        return None
    return (branched - noise) / headroom
