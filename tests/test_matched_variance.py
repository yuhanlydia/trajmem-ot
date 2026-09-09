import numpy as np
from trajmem_ot.memory_views import matched_variance_decomposition


def test_matched_variance_decomposition_separates_additive_view_and_noise_effects():
    actions = np.asarray(
        [
            [[[-1.0]], [[1.0]]],
            [[[3.0]], [[5.0]]],
        ],
        dtype=np.float64,
    )
    result = matched_variance_decomposition(actions)
    np.testing.assert_allclose(result.total, 5.0)
    np.testing.assert_allclose(result.memory_main, 4.0)
    np.testing.assert_allclose(result.noise_main, 1.0)
    np.testing.assert_allclose(result.interaction, 0.0, atol=1e-12)
    np.testing.assert_allclose(
        result.total,
        result.memory_main + result.noise_main + result.interaction,
        atol=1e-12,
    )


def test_matched_variance_decomposition_does_not_mislabel_shared_noise_as_memory():
    actions = np.asarray(
        [
            [[[0.0]], [[2.0]], [[-1.0]]],
            [[[0.0]], [[2.0]], [[-1.0]]],
            [[[0.0]], [[2.0]], [[-1.0]]],
        ],
        dtype=np.float64,
    )
    result = matched_variance_decomposition(actions)
    np.testing.assert_allclose(result.memory_main, 0.0, atol=1e-12)
    np.testing.assert_allclose(result.interaction, 0.0, atol=1e-12)
    np.testing.assert_allclose(result.noise_main, result.total)


def test_matched_variance_decomposition_identifies_interaction_only_effect():
    actions = np.asarray(
        [
            [[[1.0]], [[-1.0]]],
            [[[-1.0]], [[1.0]]],
        ],
        dtype=np.float64,
    )
    result = matched_variance_decomposition(actions)
    np.testing.assert_allclose(result.memory_main, 0.0, atol=1e-12)
    np.testing.assert_allclose(result.noise_main, 0.0, atol=1e-12)
    np.testing.assert_allclose(result.interaction, result.total)
