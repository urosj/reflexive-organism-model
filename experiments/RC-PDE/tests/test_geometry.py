import numpy as np

from rc.field import CoherenceField
from rc.geometry import CoherenceTensor, compute_K, laplace_beltrami, metric_from_K


def test_metric_positive_and_identity_for_constant_field():
    grid_shape = (20, 18)
    dx = 0.1
    field = CoherenceField(grid_shape, dx, init_func=lambda x, y: np.ones_like(x))

    K = compute_K(field, lam=2.0, xi=0.3)
    g, g_inv, sqrt_det = metric_from_K(field, K, lam=2.0, xi=0.3)

    assert g.shape == (*grid_shape, 2, 2)
    assert g_inv.shape == (*grid_shape, 2, 2)
    assert sqrt_det.shape == grid_shape
    assert np.all(sqrt_det > 0)

    # With zero gradients, metric reduces to identity regardless of λ.
    assert np.allclose(g[..., 0, 0], 1.0)
    assert np.allclose(g[..., 1, 1], 1.0)
    assert np.allclose(g[..., 0, 1], 0.0)
    assert np.allclose(g[..., 1, 0], 0.0)


def test_laplace_beltrami_matches_euclidean_when_xi_zero():
    grid_shape = (32, 32)
    dx = 0.2
    field = CoherenceField(grid_shape, dx, init_func=lambda x, y: x + y + 1.0)

    # xi=0 => K = λ C I, metric becomes identity
    K = compute_K(field, lam=1.0, xi=0.0)
    g, g_inv, sqrt_det = metric_from_K(field, K, lam=1.0, xi=0.0)

    lb = laplace_beltrami(field, g, g_inv, sqrt_det)
    # Linear field on identity metric should yield (near) zero LB
    assert np.allclose(lb, 0.0, atol=1e-8)


def test_metric_identity_uniform_field_zero_xi():
    grid_shape = (8, 8)
    dx = 0.5
    field = CoherenceField(grid_shape, dx, init_func=lambda x, y: np.ones_like(x))
    K = compute_K(field, lam=2.5, xi=0.0)
    g, g_inv, sqrt_det = metric_from_K(field, K, lam=2.5, xi=0.0)
    assert np.allclose(g[..., 0, 0], 1.0, atol=1e-6)
    assert np.allclose(g[..., 1, 1], 1.0, atol=1e-6)
    assert np.allclose(g[..., 0, 1], 0.0, atol=1e-6)
    assert np.allclose(g_inv[..., 0, 0], 1.0, atol=1e-6)
    assert np.allclose(sqrt_det, 1.0, atol=1e-6)


def test_coherence_tensor_as_array():
    ny, nx = 4, 5
    Kxx = np.ones((ny, nx))
    Kxy = np.zeros((ny, nx)) + 2.0
    Kyy = np.ones((ny, nx)) * 3.0
    K = CoherenceTensor(Kxx=Kxx, Kxy=Kxy, Kyy=Kyy)

    arr = K.as_array()
    assert arr.shape == (ny, nx, 2, 2)
    assert np.allclose(arr[..., 0, 0], Kxx)
    assert np.allclose(arr[..., 0, 1], Kxy)
    assert np.allclose(arr[..., 1, 0], Kxy)
    assert np.allclose(arr[..., 1, 1], Kyy)
