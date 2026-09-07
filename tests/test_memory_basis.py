import numpy as np
import pytest

from trajmem_ot.memory_basis import (
    combine_basis,
    history_difference_basis,
    normalize_basis,
    random_rank_one_basis,
)


def test_random_rank_one_basis_is_deterministic_unit_norm_and_rank_one():
    first = random_rank_one_basis((6, 5), count=4, seed=7)
    second = random_rank_one_basis((6, 5), count=4, seed=7)
    np.testing.assert_allclose(first, second)
    np.testing.assert_allclose(np.linalg.norm(first.reshape(4, -1), axis=1), 1.0, atol=1e-6)
    assert all(np.linalg.matrix_rank(direction, tol=1e-6) == 1 for direction in first)


def test_history_difference_basis_returns_orthonormal_principal_directions():
    rng = np.random.default_rng(11)
    left = rng.normal(size=(4, 3))
    right = rng.normal(size=(4, 3))
    differences = np.stack(
        [
            3.0 * left + 0.1 * rng.normal(size=(4, 3)),
            2.0 * left + 0.1 * rng.normal(size=(4, 3)),
            2.5 * right + 0.1 * rng.normal(size=(4, 3)),
            1.5 * right + 0.1 * rng.normal(size=(4, 3)),
        ]
    )
    basis = history_difference_basis(differences, count=2)
    flat = basis.reshape(2, -1)
    np.testing.assert_allclose(flat @ flat.T, np.eye(2), atol=1e-6)
    assert basis.shape == (2, 4, 3)


def test_combine_basis_reconstructs_linear_update():
    basis = np.zeros((2, 2, 2), dtype=np.float32)
    basis[0, 0, 0] = 1.0
    basis[1, 1, 1] = 1.0
    update = combine_basis(basis, np.asarray([2.0, -3.0]))
    np.testing.assert_array_equal(update, np.asarray([[2.0, 0.0], [0.0, -3.0]]))


def test_normalize_basis_rejects_zero_direction():
    with pytest.raises(ValueError, match="zero-norm"):
        normalize_basis(np.zeros((2, 3, 4)))


def test_history_difference_basis_rejects_too_many_directions():
    with pytest.raises(ValueError, match="count"):
        history_difference_basis(np.ones((2, 3, 4)), count=3)
