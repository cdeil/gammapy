Where to stick your Spectral Points?
====================================

The `gammapy.spectrum` module offers a number of options for positioning data points within  an energy band. This example offers a comparison between
the log center and Lafferty & Wyatt (described in [Lafferty1994]_) methods. See `gammapy.spectrum.compute_differential_flux_points` for documentation on usage.

Lafferty & Wyatt vs. Log Center
-------------------------------

In measurements made of variables whose frequency distributions do not vary rapidly, it is usually sufficient to represent the data within a bin
using the log bin center. However, it is sometimes the case where the region of interest of variables lies within a low frequency regime such that
large bin widths must be chosen to ensure statistical errors remain small. In cases where the frequency distribution varies rapidly (for
instance, exponentially) over wide bins, simply choosing the log bin center does not offer a good representation of the underlying distribution.

Instead, Lafferty & Wyatt [Lafferty1994]_ offer an alternative approach for such situations. In a bin of width :math:`\Delta x` between bounds
:math:`x_1` and :math:`x_2` for some variable :math:`x` (in spectra, this would be energy), the expectation :math:`<g_{meas}>` of the underlying
frequency distribution of :math:`x`, :math:`g(x)` is defined as  

.. math::
    <g_{meas}> = \frac{1}{\Delta x}\int_{x_1}^{x_2}{g(x) dx}

As the bin size tends to zero, the expectation of the frequency distribution tends to it's true value. The value of :math:`x` within a bin for
which the expectation should be regarded as a measurement of the true distribution is determined by Lafferty & Wyatt as the :math:`x` coordinate at
which the expectation value is equal to the mean value of the underlying frequency distribution, noted as :math:`x_{lw}`. Thus knowledge of the true frequency distribution :math:`g(x)`
or an estimate for this (determined by fitting) is required.

So it follows that, in setting expectation equal to :math:`g(x)` at :math:`x_{lw}`, the position of :math:`x_{lw}` is given by the following equation: 

.. math::
    x_{lw} = g^{-1}\left(\frac{1}{\Delta x}\int_{x_1}^{x_2}{g(x) dx}\right)
    
For instances where a power law of index -2 is taken, it can be analytically shown that the Lafferty & Wyatt method and log center method are
coincident. In the case of steeper power laws (e.g. spectral index -3), the Lafferty & Wyatt method
returns a lower :math:`x` ordinate than the log bin center, and the reverse effect is seen for less steep power laws (e.g. spectral index -1).


Power Law Assumption
--------------------

In many "real world" examples, the nature of the true underlying frequency distribution is unknown and can only be estimated, either by using a
fitting algorithm with data points or by assuming a certain spectral form. In this example, the true underlying frequency distribution
(being a piecewise power law function of indices -1 through to -5 for bins of increasing energy) is shown in blue for illustration. This would be
unknown to the user who, in this example, assumes the distribution follows a simple power law with index -4 across all energies.

The plot demonstrates that in these cases where the true frequency distribution is not known, the Lafferty & Wyatt method of positioning the data
points offers a closer representation of the underlying distribution than the log center method. Residuals showing the percentage difference of
each data point from the true distribution function are also shown.

.. plot:: tutorials/flux_point_demo.py plot_plaw
   :include-source:
