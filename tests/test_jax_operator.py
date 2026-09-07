import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from trajmem_ot.jax_operator import (
    action_jvp,
    batched_action_jvps,
    central_action_secant,
    energy_rank,
    svd_ridge_pullback,
)


def _toy_action_fn(memory):
    weights = jnp.asarray(
        [
            [0.3, -0.2, 0.5, 0.7, -0.1, 0.4],
            [-0.6, 0.8, 0.1, -0.3, 0.2, 0.5],
            [0.4, 0.1, -0.7, 0.2, 0.6, -0.2],
            [0.2, 0.5, 0.3, -0.4, 0.7, -0.6],
        ],
        dtype=memory.dtype,
    )
    return jnp.tanh(weights @ memory.reshape(-1)).reshape(2, 2)


def test_exact_jvp_matches_central_action_secant():
    memory = jnp.linspace(-0.4, 0.6, 6, dtype=jnp.float32).reshape(2, 3)
    direction = jnp.asarray([[0.2, -0.4, 0.1], [0.3, 0.5, -0.2]], dtype=jnp.float32)
    direction = direction / jnp.linalg.norm(direction)

    actions, tangent = action_jvp(_toy_action_fn, memory, direction)
    secant = central_action_secant(_toy_action_fn, memory, direction, step=1e-3)
    tangent_np = np.asarray(tangent).reshape(-1)
    secant_np = np.asarray(secant).reshape(-1)

    cosine = float(tangent_np @ secant_np / (np.linalg.norm(tangent_np) * np.linalg.norm(secant_np)))
    relative_error = float(np.linalg.norm(tangent_np - secant_np) / np.linalg.norm(tangent_np))
    assert actions.shape == (2, 2)
    assert cosine > 0.9999
    assert relative_error < 2e-4


def test_chunked_batched_jvps_match_single_direction_calls():
    memory = jnp.arange(6, dtype=jnp.float32).reshape(2, 3) / 10.0
    directions = jnp.stack(
        [
            jnp.eye(6, dtype=jnp.float32)[index].reshape(2, 3)
            for index in range(4)
        ]
    )
    actions, responses = batched_action_jvps(
        _toy_action_fn, memory, directions, chunk_size=2
    )
    expected = jnp.stack(
        [action_jvp(_toy_action_fn, memory, direction)[1] for direction in directions]
    )
    np.testing.assert_allclose(np.asarray(responses), np.asarray(expected), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(actions), np.asarray(_toy_action_fn(memory)), rtol=1e-6, atol=1e-6)


def test_svd_ridge_pullback_reduces_action_target_residual():
    response = np.asarray(
        [
            [3.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 0.1],
        ]
    )
    target = np.asarray([3.0, 2.0, 1.0])
    result = svd_ridge_pullback(response, target, damping=1e-6, rank=2)
    assert result.rank == 2
    assert result.residual_norm < np.linalg.norm(target)
    np.testing.assert_allclose(result.prediction[:2], target[:2], atol=1e-5)
    assert abs(result.coefficients[2]) < 1e-12


def test_energy_rank_returns_smallest_rank_reaching_threshold():
    singular_values = np.asarray([4.0, 3.0, 1.0])
    assert energy_rank(singular_values, threshold=0.8) == 2
    assert energy_rank(singular_values, threshold=1.0) == 3


def test_operator_rejects_bad_direction_shape():
    memory = jnp.zeros((2, 3), dtype=jnp.float32)
    with pytest.raises(ValueError, match="direction shape"):
        action_jvp(_toy_action_fn, memory, jnp.zeros((6,), dtype=jnp.float32))
