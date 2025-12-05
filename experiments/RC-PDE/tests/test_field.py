import numpy as np

from rc.field import CoherenceField


def test_constant_field_grad_laplacian_zero():
    grid_shape = (32, 40)
    dx = 0.1
    field = CoherenceField(grid_shape, dx, init_func=lambda x, y: np.ones_like(x))

    cx, cy = field.grad()
    lap = field.laplacian()

    assert cx.shape == grid_shape
    assert cy.shape == grid_shape
    assert lap.shape == grid_shape
    assert np.allclose(cx, 0.0)
    assert np.allclose(cy, 0.0)
    assert np.allclose(lap, 0.0)

    area = grid_shape[0] * grid_shape[1] * dx**2
    assert np.isclose(field.total_coherence(), area)


def test_laplace_beltrami_constant_metric():
    grid_shape = (16, 16)
    dx = 0.2

    def metric(x, y):  # noqa: ARG001
        g = np.array([[2.0, 0.1], [0.1, 3.0]])
        return np.broadcast_to(g, (*x.shape, 2, 2))

    field = CoherenceField(grid_shape, dx, init_func=lambda x, y: x * 0 + 1.0)
    lap = field.laplacian(metric=metric)
    assert lap.shape == grid_shape
    # Constant field should still yield zero Laplace-Beltrami
    assert np.allclose(lap, 0.0)

    det = 2 * 3 - 0.1 * 0.1
    expected = (grid_shape[0] * grid_shape[1] * dx**2 * np.sqrt(det))
    assert np.isclose(field.total_coherence(metric=metric), expected)


def test_hessian_shapes_and_nonzero_gaussian():
    grid_shape = (30, 30)
    field = CoherenceField(grid_shape, dx=0.1)
    hess = field.hessian()
    assert hess.shape == (2, 2, *grid_shape)
    # Gaussian bump should have some non-zero curvature
    assert np.abs(hess).sum() > 0.0


def test_sinusoidal_grad_laplacian_matches_analytic():
    nx, ny = 40, 36
    dx = 0.1
    Lx = nx * dx  # align with periodic stencil
    Ly = ny * dx

    def init_func(x, y):
        return np.sin(2 * np.pi * x / Lx) * np.sin(2 * np.pi * y / Ly)

    field = CoherenceField((ny, nx), dx, init_func=init_func)
    lap_num = field.laplacian()

    C = field.C
    coeff = -((2 * np.pi / Lx) ** 2 + (2 * np.pi / Ly) ** 2)
    lap_exact = coeff * C
    assert np.allclose(lap_num, lap_exact, atol=5e-2, rtol=5e-2)
