import numpy as np
import pytest

from trajmem_ot.memory_views import (
    allocation_grid,
    build_memory_views,
    nearest_target_coverage,
    variance_decomposition,
)


def test_build_memory_views_respects_relative_trust_radius():
    memory = np.ones((4, 3), dtype=np.float32)
    basis = np.zeros((2, 4, 3), dtype=np.float32)
    basis[0, 0, 0] = 1.0
    basis[1, 1, 1] = 1.0
    coefficients = np.asarray([[100.0, 0.0], [0.0, -100.0], [0.0, 0.0]])
    result = build_memory_views(memory, basis, coefficients, relative_radius=0.01)
    assert result.views.shape == (3, 4, 3)
    assert result.deltas.shape == (3, 4, 3)
    delta_norms = np.linalg.norm(result.deltas.reshape(3, -1), axis=1)
    assert np.all(delta_norms <= 0.010001 * np.linalg.norm(memory))
    np.testing.assert_array_equal(result.views[2], memory)


def test_variance_decomposition_obeys_law_of_total_variance():
    # Two memory views with large mean separation and small within-view noise.
    actions = np.asarray(
        [
            [[[0.0]], [[0.2]]],
            [[[4.0]], [[4.2]]],
        ],
        dtype=np.float64,
    )
    result = variance_decomposition(actions)
    assert result.between_memory > result.within_noise
    np.testing.assert_allclose(
        result.total, result.within_noise + result.between_memory, atol=1e-12
    )


def test_nearest_target_coverage_reports_view_and_overall_coverage():
    actions = np.asarray(
        [
            [[[3.0, 3.0]], [[2.0, 2.0]]],
            [[[0.1, 0.1]], [[4.0, 4.0]]],
        ]
    )
    result = nearest_target_coverage(actions, np.zeros((1, 2)), tolerance=0.2)
    assert result.covered is True
    assert result.covered_views == 1
    assert result.total_views == 2
    assert result.covered_samples == 1
    assert result.min_distance < 0.2


def test_allocation_grid_matches_requested_total_compute():
    assert allocation_grid(32, memory_view_counts=(1, 4, 8, 32)) == (
        (1, 32),
        (4, 8),
        (8, 4),
        (32, 1),
    )
    with pytest.raises(ValueError, match="divide"):
        allocation_grid(30, memory_view_counts=(4,))


def test_hypothesis_coefficients_include_base_and_symmetric_axes():
    from trajmem_ot.memory_views import hypothesis_coefficients

    coefficients = hypothesis_coefficients(view_count=5, basis_size=3, seed=9)
    np.testing.assert_array_equal(coefficients[0], np.zeros(3))
    np.testing.assert_array_equal(coefficients[1], -coefficients[2])
    np.testing.assert_array_equal(coefficients[3], -coefficients[4])
    np.testing.assert_allclose(np.linalg.norm(coefficients[1:], axis=1), 1.0)
