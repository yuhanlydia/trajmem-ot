import numpy as np
import pytest

from trajmem_ot.self_teacher import distillation_targets


def test_paired_residual_preserves_noise_correspondence():
    student = np.array([0., 10.]).reshape(2, 1, 1)
    teacher = np.array([2., 12.]).reshape(2, 1, 1)
    np.testing.assert_allclose(distillation_targets(student, teacher, mode='paired').ravel(), [2., 2.])


def test_weighted_centroid_is_a_separate_target():
    student = np.array([0., 10.]).reshape(2, 1, 1)
    teacher = np.array([2., 12.]).reshape(2, 1, 1)
    result = distillation_targets(student, teacher, mode='centroid', teacher_weights=[.25, .75])
    np.testing.assert_allclose(result.ravel(), [9.5, -.5])


def test_ot_preserves_separated_identical_modes_instead_of_collapsing_them():
    actions = np.array([-10., 10.]).reshape(2, 1, 1)
    result = distillation_targets(actions, actions, mode='ot', epsilon=.01)
    np.testing.assert_allclose(result, 0., atol=1e-9)


def test_ot_respects_teacher_mass_for_a_single_source_particle():
    student = np.array([0.]).reshape(1, 1, 1)
    teacher = np.array([2., 12.]).reshape(2, 1, 1)
    result = distillation_targets(student, teacher, mode='ot', teacher_weights=[.25, .75], epsilon=10.)
    np.testing.assert_allclose(result.ravel(), [9.5], atol=1e-7)


@pytest.mark.parametrize('weights', [[-1, 2], [0, 0], [np.nan, 1]])
def test_invalid_teacher_weights_are_rejected(weights):
    with pytest.raises(ValueError):
        distillation_targets(np.zeros((2, 1, 1)), np.ones((2, 1, 1)), mode='centroid', teacher_weights=weights)


def test_paired_requires_matching_particle_counts():
    with pytest.raises(ValueError):
        distillation_targets(np.zeros((2, 1, 1)), np.ones((3, 1, 1)), mode='paired')


def test_nonfinite_actions_are_rejected():
    with pytest.raises(ValueError):
        distillation_targets(np.array([[[np.nan]]]), np.zeros((1, 1, 1)), mode='paired')
