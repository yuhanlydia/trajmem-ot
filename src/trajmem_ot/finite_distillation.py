"""Iterative finite-response repair with deployment quantization and held-out rollback.

Callbacks keep this implementation independent of the policy/backbone. Only the
small action-response matrix is factorized; memory tensors are never decomposed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from collections import Counter
import numpy as np

from .self_teacher import distillation_targets
from .repair_acceptance import assess_repair


@dataclass(frozen=True)
class NoisePartition:
    optimize: tuple[int, ...]
    validation: tuple[int, ...]
    evaluation: tuple[int, ...]

    def __post_init__(self):
        groups = (self.optimize, self.validation, self.evaluation)
        flat = [seed for group in groups for seed in group]
        if (any(not group for group in groups) or len(set(flat)) != len(flat)
                or any(not isinstance(s, (int, np.integer)) or s < 0 for s in flat)):
            raise ValueError("noise partitions must be nonempty, unique, disjoint nonnegative integer seeds")


@dataclass(frozen=True)
class RepairConfig:
    iterations: int = 2
    rank: int = 4
    damping: float = 1e-4
    probe_relative_size: float = 2.5e-4
    step_relative_radius: float = 1.25e-3
    total_relative_radius: float = 2.5e-3
    alphas: tuple[float, ...] = (.25, .5, 1.)
    minimum_improving_fraction: float = .5
    target: str = "paired"
    ot_epsilon: float | None = None

    def __post_init__(self):
        values = (self.damping, self.probe_relative_size,
                  self.step_relative_radius, self.total_relative_radius)
        if (self.iterations < 1 or self.rank < 1
                or not np.isfinite(values).all() or any(v <= 0 for v in values)
                or not self.alphas or not np.isfinite(self.alphas).all()
                or any(a <= 0 for a in self.alphas)
                or not 0 <= self.minimum_improving_fraction <= 1
                or self.target not in ('paired', 'centroid', 'ot')):
            raise ValueError("invalid repair configuration")


@dataclass
class RepairResult:
    memory: np.ndarray
    trace: list[dict]
    evaluation: dict
    query_counts: dict[str, int]


def repair_memory(student_memory, basis, student_action, teacher_action, quantize,
                  noises: NoisePartition, config: RepairConfig, *, teacher_weights=None):
    """Fit stacked same-noise responses, validate candidates, then evaluate once.

    Memory is stored in float32 with exactly the values returned by ``quantize``.
    Radius limits apply to the actual rounded displacement relative to the initial
    memory norm. Probes use the current memory norm and unit basis directions.
    """
    initial = np.asarray(student_memory, dtype=np.float32)
    if not initial.size or not np.isfinite(initial).all():
        raise ValueError("student memory must be finite and nonempty")
    shape = initial.shape
    def q(value):
        rounded = np.asarray(quantize(np.asarray(value, dtype=np.float32)), dtype=np.float32)
        if rounded.shape != shape or not np.isfinite(rounded).all():
            raise ValueError("quantizer changed shape or returned nonfinite values")
        return rounded.copy()
    initial = q(initial)
    current = initial.copy()
    directions = np.asarray(basis, dtype=np.float32)
    if directions.ndim != initial.ndim + 1 or directions.shape[1:] != shape or not len(directions):
        raise ValueError("basis must have shape [directions, *memory_shape]")
    norms = np.linalg.norm(directions.reshape(len(directions), -1), axis=1)
    if not np.isfinite(directions).all() or np.any(norms <= 0):
        raise ValueError("basis directions must be finite and nonzero")
    directions = directions / norms.reshape((-1,) + (1,) * initial.ndim)
    reference_norm = float(np.linalg.norm(initial))
    if reference_norm <= 0:
        raise ValueError("relative-radius repair requires nonzero initial memory norm")
    step_limit = config.step_relative_radius * reference_norm
    total_limit = config.total_relative_radius * reference_norm
    counts = Counter()
    teacher_cache = {}
    def teacher(seed):
        if seed not in teacher_cache:
            teacher_cache[seed] = np.asarray(teacher_action(seed), dtype=np.float64)
            counts['teacher'] += 1
        return teacher_cache[seed]
    def actions(memory, seeds, category):
        result = []
        for seed in seeds:
            result.append(np.asarray(student_action(memory, seed), dtype=np.float64))
            counts[category] += 1
        result = np.stack(result)
        if result.ndim != 3 or not result.size or not np.isfinite(result).all():
            raise ValueError("policy must return finite [horizon, action_dim] actions")
        return result
    def teachers(seeds):
        result = np.stack([teacher(seed) for seed in seeds])
        if result.ndim != 3 or not result.size or not np.isfinite(result).all():
            raise ValueError("teacher must return finite [horizon, action_dim] actions")
        return result
    def losses(actual, target):
        if actual.shape != target.shape:
            raise ValueError("student and teacher action shapes differ")
        return np.mean((actual - target) ** 2, axis=(1, 2))
    opt_teacher = teachers(noises.optimize)
    val_teacher = teachers(noises.validation)
    trace = []
    for iteration in range(config.iterations):
        before_opt = actions(current, noises.optimize, 'optimization_baseline')
        before_val = losses(actions(current, noises.validation, 'validation_baseline'), val_teacher)
        target = distillation_targets(before_opt, opt_teacher, mode=config.target,
                                      teacher_weights=teacher_weights, epsilon=config.ot_epsilon).ravel()
        h = config.probe_relative_size * float(np.linalg.norm(current))
        columns, chords = [], []
        chord_norms = []
        for direction in directions:
            plus, minus = q(current + h * direction), q(current - h * direction)
            chord = (plus - minus) / 2.
            norm = float(np.linalg.norm(chord))
            response = (actions(plus, noises.optimize, 'finite_response_probes')
                        - actions(minus, noises.optimize, 'finite_response_probes')) / 2.
            chord_norms.append(norm)
            if norm > 0:
                chords.append(chord / norm)
                columns.append(response.ravel() / norm)
        proposal = np.zeros_like(current)
        singular_values = []
        if columns:
            operator = np.stack(columns, axis=1)
            u, singular, vh = np.linalg.svd(operator, full_matrices=False)
            rank = min(config.rank, len(singular))
            coeff = vh[:rank].T @ ((singular[:rank] / (singular[:rank]**2 + config.damping))
                                   * (u[:, :rank].T @ target))
            proposal = np.tensordot(coeff, np.stack(chords), axes=1).astype(np.float32)
            singular_values = singular.tolist()
        proposal_norm = float(np.linalg.norm(proposal))
        if proposal_norm > step_limit:
            proposal *= step_limit / proposal_norm
        candidates = []
        best = None
        for alpha in config.alphas:
            # Backtrack after rounding: pre-cast norm bounds alone are insufficient.
            scale = float(alpha)
            candidate = current
            for _ in range(32):
                candidate = q(current + scale * proposal)
                if (np.linalg.norm(candidate - current) <= step_limit
                        and np.linalg.norm(candidate - initial) <= total_limit):
                    break
                scale *= .5
            else:
                candidate = current.copy()
                scale = 0.
            after_val = losses(actions(candidate, noises.validation, 'validation_candidates'), val_teacher)
            acceptance = assess_repair(before_val, after_val,
                                       minimum_improving_fraction=config.minimum_improving_fraction)
            evidence = {'alpha': float(alpha), 'effective_alpha': scale, **asdict(acceptance),
                        'mean_loss': float(after_val.mean())}
            candidates.append(evidence)
            if acceptance.accepted and (best is None or evidence['mean_loss'] < best[0]['mean_loss']):
                best = (evidence, candidate)
        previous = current
        if best is not None:
            current = best[1]
        trace.append({'iteration': iteration + 1, 'accepted': best is not None,
                      'chosen_alpha': None if best is None else best[0]['effective_alpha'],
                      'active_directions': len(columns), 'chord_norms': chord_norms,
                      'singular_values': singular_values, 'candidates': candidates,
                      'applied_step_norm': float(np.linalg.norm(current - previous)),
                      'applied_total_norm': float(np.linalg.norm(current - initial))})
    eval_teacher = teachers(noises.evaluation)
    baseline = losses(actions(initial, noises.evaluation, 'evaluation'), eval_teacher)
    repaired = losses(actions(current, noises.evaluation, 'evaluation'), eval_teacher)
    # Zero headroom and zero change is zero recovery, not a perfect repair.
    recovery = (np.sqrt(baseline) - np.sqrt(repaired)) / (np.sqrt(baseline) + 1e-12)
    evaluation = {'baseline_losses': baseline.tolist(), 'repaired_losses': repaired.tolist(),
                  'baseline_loss_mean': float(baseline.mean()), 'repaired_loss_mean': float(repaired.mean()),
                  'recovery': recovery.tolist(), 'recovery_mean': float(recovery.mean()),
                  'noise_seeds': list(noises.evaluation),
                  'repair_headroom_present': bool(np.any(baseline > 1e-24))}
    counts['total'] = sum(counts.values())
    return RepairResult(current, trace, evaluation, dict(counts))
