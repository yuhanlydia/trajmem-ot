import numpy as np
from trajmem_ot.transport import return_tilted_transport, sinkhorn


def test_numpy_sinkhorn_matches_requested_marginals():
    p = np.asarray([0.5, 0.5])
    q = np.asarray([0.2, 0.8])
    cost = np.asarray([[0.0, 1.0], [1.0, 0.0]])
    coupling = sinkhorn(p, q, cost, epsilon=0.2, iterations=200)
    np.testing.assert_allclose(coupling.sum(axis=1), p, atol=1e-6)
    np.testing.assert_allclose(coupling.sum(axis=0), q, atol=1e-6)


def test_return_tilted_transport_moves_low_return_particle_toward_high_return_particle():
    actions = np.asarray([[[0.0]], [[2.0]]], dtype=np.float64)
    returns = np.asarray([0.0, 1.0])
    result = return_tilted_transport(actions, returns, beta=4.0, epsilon=0.1)
    assert result.target_weights[1] > result.target_weights[0]
    assert result.transport[0, 0, 0] > 0
    assert abs(result.transport[1, 0, 0]) < abs(result.transport[0, 0, 0])
