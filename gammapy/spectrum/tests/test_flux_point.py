# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import print_function, division
import numpy as np
from numpy.testing import assert_allclose
from astropy.table import Table
from ..flux_point import (_x_lafferty, _integrate, _ydiff_excess_equals_expected,
                          compute_differential_flux_points,
                          _energy_lafferty_power_law)
from ..powerlaw import power_law_eval, power_law_integral_flux
import itertools
import pytest


x_methods = ['table', 'lafferty', 'log_center']
y_methods = ['power_law', 'model']


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


@pytest.mark.parametrize('x_method,y_method', itertools.product(x_methods,
                                                                y_methods))
def test_compute_differential_flux_points(x_method, y_method):
    """Iterates through the 6 different combinations of input options.

    Tests against analytical result or result from gammapy.spectrum.powerlaw.
    """
    # Define the test cases for all possible options
    energy_min = np.array([1.0, 10.0])
    energy_max = np.array([10.0, 100.0])
    spectral_index = 2
    table = Table()
    table['ENERGY_MIN'] = energy_min
    table['ENERGY_MAX'] = energy_max
    table['ENERGY'] = np.array([2.0, 20.0])
    if x_method == 'log_center':
        energy = np.sqrt(energy_min * energy_max)
    elif x_method == 'table':
        energy = table['ENERGY'].data
    # Arbitrary model (simple exponential case)
    diff_flux_model = lambda x: np.exp(x)
    # Integral of model
    int_flux_model = lambda E_min, E_max: np.exp(E_max) - np.exp(E_min)
    if y_method == 'power_law':
        if x_method == 'lafferty':
            energy = _energy_lafferty_power_law(energy_min, energy_max,
                                                spectral_index)
            # Test that this is equal to log center result, as
            # analytically expected
            desired_energy = np.sqrt(energy_min * energy_max)
            assert_allclose(energy, desired_energy, rtol=1e-6)
        desired = power_law_eval(energy, 1, spectral_index, energy)
        int_flux = power_law_integral_flux(desired, spectral_index, energy,
                                           energy_min, energy_max)
    elif y_method == 'model':
        if x_method == 'lafferty':
            energy = _x_lafferty(energy_min, energy_max, diff_flux_model)
        desired = diff_flux_model(energy)
        int_flux = int_flux_model(energy_min, energy_max)
    int_flux_err = 0.1 * int_flux
    table['INT_FLUX'] = int_flux
    table['INT_FLUX_ERR'] = int_flux_err

    result_table = compute_differential_flux_points(table, x_method,
                                 y_method, diff_flux_model,
                                 spectral_index)
    # Test energy
    actual_energy = result_table['ENERGY'].data
    desired_energy = energy
    assert_allclose(actual_energy, desired_energy, rtol=1e-3)
    # Test flux
    actual = result_table['DIFF_FLUX'].data
    assert_allclose(actual, desired, rtol=1e-3)
    # Test error
    actual = result_table['DIFF_FLUX_ERR'].data
    desired = 0.1 * result_table['DIFF_FLUX'].data
    assert_allclose(actual, desired, rtol=1e-3)
