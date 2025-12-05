import numpy as np

from rc.visualisation import RCAnimator


def test_animator_runs_funcanimation_without_gui():
    shape = (16, 16)
    animator = RCAnimator(shape, dt=0.1)
    C = np.random.rand(*shape)
    Jx = np.zeros(shape)
    Jy = np.zeros(shape)

    for i in range(3):
        animator.update(i, C + i * 0.01, Jx, Jy)

    anim = animator.animate()
    # Call the update function for the first frame to ensure it returns artists
    artists = anim._func(0)
    assert artists
