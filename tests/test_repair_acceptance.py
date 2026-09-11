import numpy as np
import pytest

from trajmem_ot.repair_acceptance import assess_repair


def test_positive_majority_and_positive_median_accept():
    result = assess_repair([2, 2, 2, 2], [1, 1, 1, 3])
    assert result.accepted
    assert result.median_improvement == 1.
    assert result.improving_fraction == .75


def test_zero_median_rolls_back_even_with_improved_individual_noises():
    assert not assess_repair([2, 2, 2, 2], [1, 1, 3, 3]).accepted


def test_required_improving_fraction_is_enforced():
    assert not assess_repair([2, 2, 2, 2], [1, 1, 1, 3], minimum_improving_fraction=1.).accepted


@pytest.mark.parametrize('before,after', [([], []), ([1], [1, 2]), ([np.nan], [1]), ([-1], [0])])
def test_malformed_losses_are_rejected(before, after):
    with pytest.raises(ValueError):
        assess_repair(before, after)
