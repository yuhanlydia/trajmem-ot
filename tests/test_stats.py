import numpy as np
from trajmem_ot.stats import exact_two_sided_sign_test, percentile_bootstrap


def test_percentile_bootstrap_returns_finite_interval_containing_estimate_for_constant_data():
    result = percentile_bootstrap(np.ones(8), resamples=100, seed=2)
    assert result.estimate == 1.0
    assert result.lower == 1.0
    assert result.upper == 1.0


def test_sign_test_counts_pairs_not_directions():
    result = exact_two_sided_sign_test(np.asarray([1.0, 2.0, -1.0, 0.0]))
    assert result["wins"] == 2
    assert result["losses"] == 1
    assert result["ties"] == 1
