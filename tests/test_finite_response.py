import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from trajmem_ot.finite_response import (
    build_finite_response_model,
    solve_finite_response_pullback,
)


def _linear_action(memory):
    flat = memory.astype(jnp.float32).reshape(-1)
    weights = jnp.arange(1, 25, dtype=jnp.float32).reshape(4, 6) / 25.0
    return (weights @ flat).reshape(2, 2)


def test_finite_response_model_matches_linear_map():
    memory = jnp.linspace(0.5, 1.0, 6, dtype=jnp.bfloat16).reshape(2, 3)
    directions = np.stack(
        [
            np.eye(6, dtype=np.float32)[0].reshape(2, 3),
            np.eye(6, dtype=np.float32)[1].reshape(2, 3),
        ]
    )
    model = build_finite_response_model(
        _linear_action,
        memory,
        directions,
        relative_radius=0.05,
        robot_action_dim=2,
    )
    assert model.response_matrix.shape == (4, 2)
    assert model.unit_directions.shape == (2, 2, 3)
    assert all(row.changed_fraction_plus > 0 for row in model.diagnostics)
    expected = np.asarray(
        jax.jvp(
            _linear_action,
            (memory,),
            (jnp.asarray(model.unit_directions[0], dtype=memory.dtype),),
        )[1],
        dtype=np.float32,
    ).reshape(-1)
    np.testing.assert_allclose(
        model.response_matrix[:, 0], expected, rtol=0.05, atol=1e-5
    )


def test_finite_response_pullback_reduces_linear_target_and_respects_radius():
    memory = jnp.linspace(0.5, 1.0, 6, dtype=jnp.bfloat16).reshape(2, 3)
    directions = np.stack(
        [np.eye(6, dtype=np.float32)[i].reshape(2, 3) for i in range(3)]
    )
    model = build_finite_response_model(
        _linear_action,
        memory,
        directions,
        relative_radius=0.05,
        robot_action_dim=2,
    )
    target = (0.02 * model.response_matrix[:, 0]).reshape(model.output_shape)
    result = solve_finite_response_pullback(
        model,
        memory,
        target,
        damping=1e-8,
        rank=3,
        max_relative_norm=0.1,
    )
    assert result.linear_residual_norm < np.linalg.norm(target)
    assert result.applied_relative_norm <= 0.1001
    assert result.quantized_memory.shape == memory.shape


def test_finite_response_rejects_erased_direction():
    memory = jnp.ones((2, 3), dtype=jnp.bfloat16)
    directions = np.ones((1, 2, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="erased"):
        build_finite_response_model(
            _linear_action,
            memory,
            directions,
            relative_radius=1e-12,
            robot_action_dim=2,
        )
