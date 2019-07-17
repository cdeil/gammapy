# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Simple image for cosmic ray spectra at Earth.

For measurements, the "Database of Charged Cosmic Rays (CRDB)" is a great resource:
http://lpsc.in2p3.fr/cosmic-rays-db/
"""

from astropy import units as u
from gammapy.modeling.models.spectrum.core import PowerLaw, SpectralLogGaussian

__all__ = ["cosmic_ray_spectrum"]


def cosmic_ray_spectrum(particle="proton"):
    """Cosmic ray flux at Earth.

    These are the spectra assumed in this CTA study:
    Table 3 in https://ui.adsabs.harvard.edu/abs/2013APh....43..171B

    The hadronic spectra are simple power-laws, the electron spectrum
    is the sum of  a power law and a log-normal component to model the
    "Fermi shoulder".

    Parameters
    ----------
    energy : `~astropy.units.Quantity`
        Particle energy
    particle : {'electron', 'proton', 'He', 'N', 'Si', 'Fe'}
        Particle type

    Returns
    -------
    flux : `~astropy.units.Quantity`
        Cosmic ray flux in unit ``m^-2 s^-1 TeV^-1 sr^-1``
    """

    if particle == "proton":
        model = PowerLaw(
            amplitude=0.096 * u.Unit("1 / (m2 s TeV sr)"),
            index=2.70,
            reference=1 * u.TeV,
        )
    elif particle == "N":
        model = PowerLaw(
            amplitude=0.0719 * u.Unit("1 / (m2 s TeV sr)"),
            index=2.64,
            reference=1 * u.TeV,
        )
    elif particle == "Si":
        model = PowerLaw(
            amplitude=0.0284 * u.Unit("1 / (m2 s TeV sr)"),
            index=2.66,
            reference=1 * u.TeV,
        )
    elif particle == "Fe":
        model = PowerLaw(
            amplitude=0.0134 * u.Unit("1 / (m2 s TeV sr)"),
            index=2.63,
            reference=1 * u.TeV,
        )
    elif particle == "electron":
        model = PowerLaw(
            amplitude=6.85e-5 * u.Unit("1 / (m2 s TeV sr)"),
            index=3.21,
            reference=1 * u.TeV,
        ) + SpectralLogGaussian(
            norm=3.19e-3 * u.Unit("1 / (m2 s sr)"), mean=0.107 * u.TeV, sigma=0.776
        )
    else:
        raise ValueError("Invalid argument for particle: {}".format(particle))

    return model
