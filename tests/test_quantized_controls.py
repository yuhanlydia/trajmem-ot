import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from trajmem_ot.finite_response import quantized_norm_matched_memory


def test_quantized_norm_matched_memory_matches_applied_bfloat16_norm():
    memory = jnp.linspace(-2.0, 2.0, 2048, dtype=jnp.bfloat16).reshape(32, 64)
    rng = np.random.default_rng(3)
    direction = rng.normal(size=memory.shape).astype(np.float32)
    target_norm = 0.75
    result = quantized_norm_matched_memory(
        memory,
        direction,
        target_norm=target_norm,
        relative_tolerance=0.02,
    )
    assert result.applied_norm > 0
    assert abs(result.applied_norm - target_norm) / target_norm <= 0.02
    np.testing.assert_allclose(
        np.asarray(result.quantized_memory, dtype=np.float32)
        - np.asarray(memory, dtype=np.float32),
        result.applied_delta,
        atol=0,
    )


def test_quantized_norm_matched_memory_rejects_zero_direction():
    memory = jnp.ones((4, 4), dtype=jnp.bfloat16)
    with pytest.raises(ValueError, match="zero-norm"):
        quantized_norm_matched_memory(memory, np.zeros((4, 4)), target_norm=1.0)
