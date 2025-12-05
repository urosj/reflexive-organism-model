import numpy as np

from rc.events import apply_spark
from rc.field import CoherenceField


def test_spark_creates_bump():
    ny, nx = 20, 20
    dx = 0.1
    # Flat field -> Hessian determinant zero everywhere
    field = CoherenceField((ny, nx), dx, init_func=lambda x, y: np.ones_like(x))
    # Slightly perturb to avoid zero division
    field.C += 1e-6

    changed = apply_spark(field, epsilon=1e-8, bump_amplitude=0.2)
    assert changed
    assert np.max(field.C) > np.mean(field.C) * 1.1
