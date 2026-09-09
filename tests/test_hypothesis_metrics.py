import numpy as np

from trajmem_ot.hypothesis_metrics import (
    compare_hypothesis_coverage,
    nearest_set_distances,
)


def test_nearest_set_distances_matches_known_geometry():
    candidates = np.asarray([[0.0, 0.0], [2.0, 0.0]])
    targets = np.asarray([[1.0, 0.0], [3.0, 0.0]])
    np.testing.assert_allclose(nearest_set_distances(candidates, targets), [1.0, 1.0])


def test_oracle_branching_improves_coverage_over_wrong_shared_memory():
    correct_reference = np.asarray([[0.0], [0.1], [-0.1]])
    correct_targets = np.asarray([[0.05], [-0.05]])
    wrong_only = np.asarray([[5.0], [4.5], [5.5]])
    branched = np.asarray([[5.0], [0.02], [-0.02]])
    correct_only = np.asarray([[0.0], [0.03]])
    result = compare_hypothesis_coverage(
        correct_reference,
        correct_targets,
        wrong_only,
        branched,
        correct_only=correct_only,
        tolerance_quantile=1.0,
        tolerance_multiplier=1.5,
    )
    assert result.noise_only.coverage_fraction == 0.0
    assert result.branched.coverage_fraction == 1.0
    assert result.correct_only is not None
    assert result.coverage_gain == 1.0
    assert result.mean_distance_improvement > 0
