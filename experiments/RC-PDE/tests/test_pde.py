import numpy as np

from rc.field import CoherenceField
from rc.pde import step_euler


def test_short_explicit_run_conserves_coherence_and_nonnegative():
    grid_shape = (32, 32)
    dx = 0.1
    field = CoherenceField(grid_shape, dx)

    target = field.total_coherence()

    for _ in range(10):
        step_euler(field, dt=0.01, target_coherence=target)
        assert field.total_coherence() == field.total_coherence()  # sanity: finite

    final_total = field.total_coherence()
    assert np.isclose(final_total, target, rtol=1e-3, atol=1e-3)
    assert np.all(field.C >= 0)
