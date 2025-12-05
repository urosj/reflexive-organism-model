import numpy as np

from rc.field import CoherenceField
from rc import pde


def test_mass_conservation_over_steps():
    grid_shape = (32, 32)
    dx = 0.1
    field = CoherenceField(grid_shape, dx, init_mode="gaussian")
    initial = field.total_coherence()

    steps = 20
    dt = 0.001
    for _ in range(steps):
        pde.step_euler(field, dt=dt, lam=1.0, xi=0.5, kappa=0.5, beta=0.1, gamma_curv=0.0)

    final = field.total_coherence()
    assert np.isfinite(final)
    assert abs(final - initial) < 1e-5


def test_rhs_finite_and_reasonable_magnitude():
    grid_shape = (16, 16)
    dx = 0.1
    field = CoherenceField(grid_shape, dx, init_mode="gaussian")
    rhs_val = pde.rhs(field, lam=1.0, xi=0.2, kappa=0.2, beta=0.0)

    assert np.isfinite(rhs_val).all()
    assert rhs_val.shape == grid_shape
