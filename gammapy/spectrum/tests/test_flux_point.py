# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import print_function, division
import numpy as np
from numpy.testing import assert_allclose
from astropy.table import Table
from ..flux_point import _x_lafferty, _integrate, _ydiff_excess_equals_expected, compute_differential_flux_points, _energy_lafferty_power_law


def test_x_lafferty():
    """Tests Lafferty & Wyatt x-point method.

    Using input function g(x) = 10^4 exp(-6x) against 
    check values from paper Lafferty & Wyatt. Nucl. Instr. and Meth. in Phys.
    Res. A 355 (1995) 541-547, p. 542 Table 1
    """
    # These are the results from the paper
    desired = np.array([0.048, 0.190, 0.428, 0.762])

    f = lambda x: (10 ** 4) * np.exp(-6 * x)
    emins = np.array([0.0, 0.1, 0.3, 0.6])
    emaxs = np.array([0.1, 0.3, 0.6, 1.0])
    actual = _x_lafferty(xmin=emins, xmax=emaxs, function=f)
    assert_allclose(actual, desired, atol=1e-3)


def test_integration():
    function = lambda x: x ** 2
    xmin = np.array([-2])
    xmax = np.array([2])
    indef_int = lambda x: (x ** 3) / 3
    # Calculate analytical result
    desired = indef_int(xmax) - indef_int(xmin)
    # Get numerical result
    actual = _integrate(xmin, xmax, function, segments=1e3)
    # Compare, bounds suitable for number of segments
    assert_allclose(actual, desired, rtol=1e-2)


def test_ydiff_excess_equals_expected():
    """Tests y-value normalization adjustment method.
    """
    model = lambda x: x ** 2
    xmin = np.array([10, 20, 30, 40])
    xmax = np.array([20, 30, 40, 50])
    yint = np.array([42, 52, 62, 72])  # 'True' integral flux in this test bin
    # Get values
    x_values = np.array(_x_lafferty(xmin, xmax, model))
    y_values = _ydiff_excess_equals_expected(yint, xmin, xmax, x_values, model)
    # Set up test case comparison
    y_model = model(np.array(x_values))
    # Test comparison result
    desired = _integrate(xmin, xmax, model)
    # Test output result
    actual = y_model * (yint / y_values)
    # Compare
    assert_allclose(actual, desired, rtol=1e-6)


def test_compute_differential_flux_points():
    # Test data
    desired = [1.81535907e-11, 3.65324908e-11, 4.91195774e-11, 7.08774745e-11]
    int_fluxes = [3.63022297799e-11, 2.55718656562e-10, 9.82378126447e-10,
                  4.96140616902e-09]
    int_flux_errors = [3.63022297799e-12, 2.55718656562e-11, 9.82378126447e-11,
                       4.96140616902e-10]
    emins = [1, 3, 10, 30]
    emaxs = [3, 10, 30, 100]
    # Create test table
    table = Table()
    table['ENERGY_MIN'] = emins
    table['ENERGY_MAX'] = emaxs
    table['INT_FLUX'] = int_fluxes
    table['INT_FLUX_ERR'] = int_flux_errors
    power_law = compute_differential_flux_points(table=table,
                                                 spectral_index=2.3,
                                                 x_method='lafferty',
                                                 y_method='power_law')
    actual = power_law['DIFF_FLUX']
    # Test flux
    assert_allclose(actual, desired, rtol=1e-2)
    # Test error
    actual = power_law['DIFF_FLUX_ERR']
    desired = 0.1 * power_law['DIFF_FLUX']
    assert_allclose(actual, desired, rtol=1e-6)


def test_energy_lafferty_power_law():
    """Checks spectral index = -2 gives same result as log bin center.
    """
    energy_min = 10
    energy_max = 100
    spectral_index = -2
    desired = np.sqrt(energy_min * energy_max)
    actual = _energy_lafferty_power_law(energy_min, energy_max, spectral_index)
    assert_allclose(actual, desired, rtol=1e-6)
