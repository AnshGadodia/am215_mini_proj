import numpy as np
from am215_mini_proj.models import BernoulliRandomWalk


def test_all_up_days():
    model = BernoulliRandomWalk(
        p=1.0, delta_up=0.01, delta_down=-0.01
    )
    paths = model.simulate(s0=100, n_days=3, n_paths=5)

    expected = 100 * 1.01 ** np.arange(4)
    np.testing.assert_allclose(
        paths, np.repeat(expected[:, None], 5, axis=1)
    )