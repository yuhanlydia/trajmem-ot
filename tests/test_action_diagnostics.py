import numpy as np
import pytest

from trajmem_ot.action_diagnostics import action_slice_metrics, primal_delta_norm, vector_cosine


def test_action_slice_metrics_separates_robot_and_padded_channels():
    jvp = np.zeros((2, 32), dtype=np.float32)
    chord = np.zeros((2, 32), dtype=np.float32)
    jvp[:, :8] = 1.0
    chord[:, :8] = 2.0
    jvp[:, 8:] = 1.0
    chord[:, 8:] = -1.0

    result = action_slice_metrics(jvp, chord)

    assert result["cosine_robot_8d"] == pytest.approx(1.0)
    assert result["cosine_padded_24d"] == pytest.approx(-1.0)
    assert result["cosine_all_32d"] < 0.0
    assert result["jvp_norm_robot_8d"] == pytest.approx(4.0)


def test_primal_delta_norm_uses_float64_accumulation():
    assert primal_delta_norm(np.array([1.0, 2.0]), np.array([4.0, 6.0])) == pytest.approx(5.0)


def test_vector_cosine_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        vector_cosine(np.ones(2), np.ones(3))
