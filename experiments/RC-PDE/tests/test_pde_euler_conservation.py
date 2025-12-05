import numpy as np

from rc.field import CoherenceField
from rc.pde import step_euler


def test_euler_conserves_total_coherence_over_longer_run():
    grid_shape = (20, 20)
    dx = 0.1
    field = CoherenceField(grid_shape, dx)
    target = field.total_coherence()

    steps = 100
    dt = 0.001
    for _ in range(steps):
        step_euler(field, dt=dt, target_coherence=target)

    total = field.total_coherence()
    assert abs(total - target) < 1e-5
    assert np.all(field.C >= 0)
