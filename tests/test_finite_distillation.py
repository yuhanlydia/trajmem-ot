import numpy as np
import pytest
from trajmem_ot.finite_distillation import NoisePartition, RepairConfig, repair_memory

NOISES = NoisePartition((1, 2), (3, 4), (5, 6))

def run(iterations=2, **kwargs):
    config = RepairConfig(iterations=iterations, rank=1, damping=1e-12,
                          probe_relative_size=.01, step_relative_radius=10.,
                          total_relative_radius=10., alphas=(1.,))
    return repair_memory(np.array([1.]), np.array([[1.]]),
                         kwargs.pop('student_action', lambda m, s: (m*m).reshape(1, 1)),
                         kwargs.pop('teacher_action', lambda s: np.array([[4.]])),
                         kwargs.pop('quantize', lambda m: m), NOISES, config, **kwargs)


def test_nonlinear_repair_relinearizes_at_accepted_memory():
    one = run(1)
    two = run(2)
    np.testing.assert_allclose(one.memory, [2.5], atol=1e-5)
    np.testing.assert_allclose(two.memory, [2.05], atol=1e-5)
    assert two.evaluation['repaired_loss_mean'] < one.evaluation['repaired_loss_mean']
    assert two.query_counts['finite_response_probes'] == 8
    assert all(row['accepted'] for row in two.trace)


def test_validation_can_reject_every_step_without_mutating_input():
    result = run(teacher_action=lambda s: np.array([[4. if s < 3 else 1.]]))
    np.testing.assert_array_equal(result.memory, [1.])
    assert not any(row['accepted'] for row in result.trace)


def test_evaluation_noises_are_never_seen_before_final_evaluation():
    calls = []
    def action(m, s):
        calls.append(s)
        return (m*m).reshape(1, 1)
    run(student_action=action)
    first = next(i for i, s in enumerate(calls) if s in NOISES.evaluation)
    assert set(calls[first:]) <= set(NOISES.evaluation)


def test_seed_overlap_is_rejected():
    with pytest.raises(ValueError):
        NoisePartition((1, 2), (2, 3), (4,))


def test_quantization_erased_probes_are_a_recorded_noop():
    result = run(quantize=lambda m: np.round(m))
    np.testing.assert_array_equal(result.memory, [1.])
    assert result.trace[0]['active_directions'] == 0
    assert result.query_counts['finite_response_probes'] == 8


def test_quantized_applied_delta_respects_total_and_step_radii():
    config = RepairConfig(iterations=3, rank=1, probe_relative_size=.2,
                          step_relative_radius=.16, total_relative_radius=.25)
    result = repair_memory(np.array([1.]), np.array([[1.]]),
                           lambda m,s: m.reshape(1,1), lambda s: np.array([[3.]]),
                           lambda m: np.round(m, 1), NOISES, config)
    assert np.linalg.norm(result.memory - 1.) <= .25 + 1e-6
    assert all(row['applied_step_norm'] <= .16 + 1e-6 for row in result.trace)


def test_already_clean_memory_does_not_report_spurious_perfect_recovery():
    result = run(teacher_action=lambda s: np.array([[1.]]))
    assert result.evaluation['recovery_mean'] == 0.
    assert result.evaluation['repair_headroom_present'] is False
