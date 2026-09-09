import numpy as np
from trajmem_ot.hypothesis_metrics import (
    compare_hypothesis_coverage_curve,
    headroom_recovery,
)


def test_headroom_recovery_normalizes_gain_by_available_support_gap():
    assert headroom_recovery(noise_coverage=0.25, branched_coverage=1.0) == 1.0
    assert headroom_recovery(noise_coverage=1.0, branched_coverage=1.0) is None


def test_coverage_curve_is_monotone_in_tolerance_and_keeps_multiplier_order():
    correct_reference = np.asarray([[0.0], [0.1], [-0.1], [0.05]])
    correct_targets = np.asarray([[0.02], [-0.02], [0.08]])
    wrong = np.asarray([[2.0], [2.1], [1.9], [2.2]])
    branched = np.asarray([[2.0], [0.01], [-0.01], [0.09]])
    curve = compare_hypothesis_coverage_curve(
        correct_reference,
        correct_targets,
        wrong,
        branched,
        tolerance_multipliers=(0.75, 1.0, 1.25, 1.5),
    )
    assert [point.tolerance_multiplier for point in curve] == [0.75, 1.0, 1.25, 1.5]
    branch_coverages = [point.comparison.branched.coverage_fraction for point in curve]
    assert branch_coverages == sorted(branch_coverages)
