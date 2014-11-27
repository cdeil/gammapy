# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division

import numpy as np
from numpy.testing.utils import assert_allclose

from astropy.tests.helper import pytest
from astropy.convolution import Gaussian2DKernel


from ...detect import compute_ts_map
from ...datasets import load_poisson_stats_image


try:
    import scipy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@pytest.mark.skipif('not HAS_SCIPY')
def test_compute_ts_map():
    """Minimal test of compute_ts_map"""
    data = load_poisson_stats_image(extra_info=True)
    kernel = Gaussian2DKernel(5)
    exposure = np.ones(data['counts'].shape) * 1E12
    result = compute_ts_map(data['counts'], data['background'], exposure,
                            kernel, debug=True)
    assert_allclose([[99], [99]], np.where(result.ts == result.ts.max()))
    assert_allclose(1714.2325342455877, result.ts[99, 99])
    assert_allclose(3, result.niter[99, 99])
    assert_allclose(1.0259579427691128e-09, result.amplitude[99, 99])