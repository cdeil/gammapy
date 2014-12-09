# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Functions to compute TS maps

This is in the exploratory phase, we are trying to get a fast tool for a large map.
Here we compare different ways to split the map into parts and different optimizers.

Reference : Stewart (2009) "Maximum-likelihood detection of sources among Poissonian noise"
            Appendix A: Cash amplitude fitting
            http://adsabs.harvard.edu/abs/2009A%26A...495..989S

TODO:

- try different optimizers
- give good fit start values
- PSF-convolved Gauss-source kernels
- Use multiple CPUs with multiprocessing.
- check that Li & Ma significance maps match sqrt(ts) maps for kernel with weights 0 / 1 only
- implement On / Off and On Likelihood fitting
- implement optimized linear filter from Stewart paper
- implement down-sampling for large kernels or generally for speed
- implement possibility to only compute part of the TS image
- understand negative amplitudes!???
- speed profiling:

  - expect speed constant with image size
  - expect speed inversely proportional to number of pixels in the kernel
  - expect speedup proportional to number of cores

- accuracy profiling:

  - want accuracy of TS = 0.1 for all regimes; no need to waste cycles on higher accuracy
  - don't care about accuracy for TS < 1

- check distribution against expected chi2(ndf) distribution
- HGPS survey sensitiviy calculation (maybe needs cluster computing?)
"""
from __future__ import print_function, division
import logging
import warnings
from itertools import product
from functools import partial

import numpy as np
from imageutils import extract_array_2d

from ..stats import cash
from ..image import disk_correlate
from ..extern.zeros import newton


__all__ = ['compute_ts_map']


FLUX_FACTOR = 1E-12
MAXNITER = 20


class TSMapResult(dict):
    """
    Represents the TS map computation result.

    Attributes
    ----------
    ts : ndarray
        Estimated TS map.
    amplitude : ndarray
        Estimated best fit flux amplitude map.
    niter : ndarray
        Number of iterations per pixel.
    runtime : float
        Time needed to compute TS map.
    """
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in self.items()])
        else:
            return self.__class__.__name__ + "()"


def f_root(x, counts, background, model):
    """
    Function to find root of. Described in Appendix A, Stewart (2009).

    Parameters
    ----------
    x : float
        Model amplitude.
    counts : array
        Count map slice, where model is defined.
    background : array
        Background map slice, where model is defined.
    model : array
        Source template (multiplied with exposure).
    """
    return (model - (counts * model) / (background + x * FLUX_FACTOR * model)).sum()


def f_cash(x, counts, background, model):
    """
    Cash fit statistics wrapper for TS map computation.

    Parameters
    ----------
    x : float
        Model amplitude.
    counts : array
        Count map slice, where model is defined.
    background : array
        Background map slice, where model is defined.
    model : array
        Source template (multiplied with exposure).
    """
    with np.errstate(invalid='ignore', divide='ignore'):
        return cash(counts, background + x * FLUX_FACTOR * model).sum()


def compute_ts_map(counts, background, exposure, kernel, flux=None,
                   method='root', optimizer='Brent', parallel=True,
                   threshold=None, debug=False):
    """
    Compute TS map using different methods.

    Parameters
    ----------
    counts : array
        Count map
    background : array
        Background map
    exposure : array
        Exposure map
    kernel : `astropy.convolution.Kernel2D`
        Source model kernel.
    flux : float (None)
        Flux map used as a starting value for the amplitude fit.
    method : str ('root')
        The following options are available:
            * ``'root'`` (default)
                Fit amplitude finding roots of the the derivative of
                the fit statistics. Described in Appendix A in Stewart (2009).
            * ``'fit scipy'``
                Use `scipy.optimize.minimize_scalar` for fitting.
            * ``'fit minuit'``
                Use minuit for fitting.
    optimizer : str ('Brent')
        Which optimizing algorithm to use from scipy. See
        `scipy.optimize.minimize_scalar` for options.
    parallel : bool (True)
        Whether to use multiple cores for parallel processing.
    threshold : float (None)
        If the TS value corresponding to the initial flux estimate is not above
        this threshold, the optimizing step is omitted to save computing time.
    debug : bool (False)
        Run function in debug mode which returns and additional fitted flux and
        number of iterations map (see section `Returns`).

    Returns
    -------
    TS : `TSMapResult`
        `TSMapResult` object.
    """
    from scipy.ndimage.morphology import binary_erosion
    from time import time
    t_0 = time()

    assert counts.shape == background.shape
    assert counts.shape == exposure.shape

    if flux is None:
        radius = _flux_correlation_radius(kernel)
        logging.info('Using correlation radius of {0:.1f} pix to estimate initial flux.'.format(radius))
        flux = disk_correlate((counts - background) / exposure, radius) / FLUX_FACTOR
    else:
        assert counts.shape == flux.shape

    TS = np.zeros(counts.shape)

    x_min, x_max = kernel.shape[1] // 2, counts.shape[1] - kernel.shape[1] // 2
    y_min, y_max = kernel.shape[0] // 2, counts.shape[0] - kernel.shape[0] // 2
    positions = product(range(x_min, x_max), range(y_min, y_max))

    # Positions where exposure == 0 and flux < 0 are not processed
    mask = binary_erosion(exposure > 0, np.ones(kernel.shape))
    positions = [(i, j) for i, j in positions if mask[j][i] and flux[j][i] > 0]
    wrap = partial(_ts_value, counts=counts, exposure=exposure,
                   background=background, kernel=kernel, flux=flux,
                   method=method, optimizer=optimizer, threshold=threshold,
                   debug=debug)

    if parallel:
        from multiprocessing import Pool, cpu_count
        logging.info('Using {0} cores to compute TS map.'.format(cpu_count()))
        pool = Pool()
        results = pool.map(wrap, positions)
        pool.close()
        pool.join()
    else:
        results = map(wrap, positions)

    # Set TS values at given positions
    i, j = zip(*positions)
    if debug:
        amplitudes = np.zeros(TS.shape)
        niter = np.zeros(TS.shape)
        TS[j, i] = [_[0] for _ in results]
        amplitudes[j, i] = [_[1] for _ in results]
        niter[j, i] = [_[2] for _ in results]
        return TSMapResult(ts=TS, amplitude=amplitudes * FLUX_FACTOR,
                           niter=niter, runtime=time() - t_0)
    else:
        TS[j, i] = results
        return TSMapResult(ts=TS)


def _ts_value(position, counts, exposure, background, kernel, flux,
              method, optimizer, threshold, debug):
    """
    Compute TS value at a given pixel position i, j using the approach described
    in Stewart (2009).

    Parameters
    ----------
    position : tuple (i, j)
        Pixel position.
    counts : array
        Count map.
    background : array
        Background map.
    exposure : array
        Exposure map.
    kernel : astropy.convolution.core.Kernel2D
        Source model kernel.
    flux : array
        Flux map. The flux value at the given pixel position is used as
        starting value for the minimization.

    Returns
    -------
    TS : float
        TS value at the given pixel position.
    """
    # Get data slices
    counts_slice = extract_array_2d(counts, kernel.shape, position).astype(float)
    background_slice = extract_array_2d(background, kernel.shape, position).astype(float)
    exposure_slice = extract_array_2d(exposure, kernel.shape, position).astype(float)
    model = (exposure_slice * kernel._array).astype(float)

    # Compute null hypothesis statistics
    flux_value = flux[position[1]][position[0]]

    with np.errstate(invalid='ignore', divide='ignore'):
        C_0 = cash(counts_slice, background_slice).sum()
    with np.errstate(invalid='ignore', divide='ignore'):
        C_1 = cash(counts_slice, background_slice + flux_value * FLUX_FACTOR * model).sum()

    # Don't fit if pixel is low significant
    TS = C_0 - C_1
    if threshold is not None and TS < threshold:
        if debug:
            return TS, flux_value, 0
        else:
            return TS
    else:
        if method == 'fit minuit':
            amplitude, niter = _fit_amplitude_minuit(counts_slice, background_slice,
                                                     model, flux_value)
        elif method == 'fit scipy':
            amplitude, niter = _fit_amplitude_scipy(counts_slice, background_slice,
                                                    model, flux_value)
        elif method == 'root':
            amplitude, niter = _root_amplitude(counts_slice, background_slice,
                                               model, flux_value)
        if niter > MAXNITER:
            logging.warn('Exceeded maximum number of function evaluations!')
            if debug:
                return np.nan, amplitude, niter
            else:
                return np.nan
        with np.errstate(invalid='ignore', divide='ignore'):
            C_1 = cash(counts_slice, background_slice + amplitude * FLUX_FACTOR * model).sum()
        # Compute and return TS value
        if debug:
            return C_0 - C_1, amplitude, niter
        else:
            return C_0 - C_1


def _root_amplitude(counts, background, model, flux):
    """
    Fit amplitude by finding roots. See Appendix A Stewart (2009).

    Parameters
    ----------
    counts : array
        Slice of count map.
    background : array
        Slice of background map.
    model : array
        Model template to fit.
    flux : float
        Starting value for the fit.

    Returns
    -------
    amplitude : float
        Fitted flux amplitude.
    niter : int
        Number of function evaluations needed for the fit.
    """
    args = (counts, background, model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return newton(f_root, flux, args=args, maxiter=MAXNITER, tol=0.1)
        except RuntimeError:
            # Where the root finding fails NaN is set as amplitude
            return np.nan, MAXNITER


def _fit_amplitude_scipy(counts, background, model, flux, optimizer='Brent'):
    """
    Fit amplitude using scipy.optimize.

    Parameters
    ----------
    counts : array
        Slice of count map.
    background : array
        Slice of background map.
    model : array
        Model template to fit.
    flux : float
        Starting value for the fit.

    Returns
    -------
    amplitude : float
        Fitted flux amplitude.
    niter : int
        Number of function evaluations needed for the fit.
    """
    from scipy.optimize import minimize_scalar
    args = (counts, background, model)
    try:
        bracket = (0, flux, 10 * flux)
        result = minimize_scalar(f_cash, bracket=bracket, args=args,
                                 method=optimizer, tol=10)
        return result.x, result.nfev
    except ValueError:
        result = minimize_scalar(f_cash, args=args, method=optimizer, tol=0.1)
        return result.x, result.nfev


def _fit_amplitude_minuit(counts, background, model, flux):
    """
    Fit amplitude using minuit.

    Parameters
    ----------
    counts : array
        Slice of count map.
    background : array
        Slice of background map.
    model : array
        Model template to fit.
    flux : float
        Starting value for the fit.

    Returns
    -------
    amplitude : float
        Fitted flux amplitude.
    niter : int
        Number of function evaluations needed for the fit.
    """
    from iminuit import Minuit

    def stat(x):
        return f_cash(x, counts, background, model)
    minuit = Minuit(f_cash, x=flux, pedantic=False, print_level=0)
    minuit.migrad()
    return minuit.values['x'], minuit.ncalls


def _flux_correlation_radius(kernel, containment=0.8):
    """
    Compute equivalent Tophat kernel radius for a given kernel instance and
    containment fraction.

    Parameters
    ----------
    kernel : `astropy.convolution.Kernel2D`
        Name of the kernel type.
    containment : float
        Containment fraction.

    Returns
    -------
    kernel : `astropy.convolution.Tophat2DKernel`
        Equivalent Tophat kernel.
    """
    from ..image.measure import measure_containment_radius
    from astropy.io.fits import ImageHDU
    kernel_image = ImageHDU(kernel.array)
    y, x = kernel.center
    # Containment radius of Tophat kernel is given by R_C = r_0 * sqrt(C)
    return (measure_containment_radius(kernel_image, x, y, containment)
            / np.sqrt(containment))
