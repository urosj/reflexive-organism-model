import numpy as np

from rc.field import CoherenceField
from rc.flux import divergence, flux, velocity
from rc.geometry import compute_K, metric_from_K


def test_velocity_zero_for_constant_field():
    grid_shape = (16, 16)
    field = CoherenceField(grid_shape, dx=0.1, init_func=lambda x, y: np.ones_like(x))
    K = compute_K(field)
    g, g_inv, sqrt_det = metric_from_K(field, K)

    vx, vy = velocity(field, g_inv, K)
    assert np.allclose(vx, 0.0)
    assert np.allclose(vy, 0.0)

    Jx, Jy = flux(field, (vx, vy))
    assert np.allclose(Jx, 0.0)
    assert np.allclose(Jy, 0.0)

    div = divergence(Jx, Jy, dx=field.dx)
    assert np.allclose(div, 0.0)

    div_cov = divergence(Jx, Jy, dx=field.dx, metric=(g, g_inv, sqrt_det))
    assert np.allclose(div_cov, 0.0)


def test_divergence_matches_finite_difference():
    dx = 0.2
    ny, nx = 20, 18
    y = np.arange(ny) * dx
    x = np.arange(nx) * dx
    X, Y = np.meshgrid(x, y, indexing="xy")

    Jx = X + 2 * Y
    Jy = -X + Y

    div = divergence(Jx, Jy, dx=dx)

    # Analytical divergence: d/dx Jx + d/dy Jy = 1 + 1 = 2
    assert np.allclose(div, 2.0)


def test_constant_field_divergence_zero():
    field = CoherenceField((10, 10), dx=0.1, init_func=lambda x, y: np.ones_like(x))
    Jx = np.ones_like(field.C) * 0.0
    Jy = np.ones_like(field.C) * 0.0
    assert np.allclose(divergence(Jx, Jy, dx=field.dx), 0.0)
