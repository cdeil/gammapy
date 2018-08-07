# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import copy
import astropy.units as u
import operator
from ..utils.modeling import ParameterList, Parameter
from ..utils.scripts import make_path
from ..maps import Map

__all__ = [
    'SourceLibrary',
    'SkyModel',
    'CompoundSkyModel',
    'SumSkyModel',
    'SkyDiffuseCube',
]


class SourceLibrary(object):
    """Collection of `~gammapy.cube.models.SkyModel`

    Parameters
    ----------
    skymodels : list of `~gammapy.cube.models.SkyModel`
        Sky models

    Examples
    --------

    Read a SourceLibrary from an XML file::

        from gammapy.cube import SourceLibrary
        filename = '$GAMMAPY_EXTRA/test_datasets/models/fermi_model.xml'
        sourcelib = SourceLibrary.from_xml(filename)
    """

    def __init__(self, skymodels):
        self.skymodels = skymodels

    @classmethod
    def from_xml(cls, xml):
        """Read SourceLibrary from XML string"""
        from ..utils.serialization import xml_to_source_library
        return xml_to_source_library(xml)

    @classmethod
    def read(cls, filename):
        """Read SourceLibrary from XML file

        The XML definition of some models is uncompatible with the models
        currently implemented in gammapy. Therefore the following modifications
        happen to the XML model definition

        * PowerLaw: The spectral index is negative in XML but positive in
          gammapy. Parameter limits are ignored

        * ExponentialCutoffPowerLaw: The cutoff energy is transferred to
          lambda = 1 / cutof energy on read
        """
        path = make_path(filename)
        xml = path.read_text()
        return cls.from_xml(xml)

    def to_xml(self, filename):
        """Write SourceLibrary to XML file"""
        from ..utils.serialization import source_library_to_xml
        xml = source_library_to_xml(self)
        filename = make_path(filename)
        with filename.open('w') as output:
            output.write(xml)

    def to_compound_model(self):
        """Return `~gammapy.cube.models.CompoundSkyModel`"""
        return np.sum([m for m in self.skymodels])

    def to_sum_model(self):
        """Return `~gammapy.cube.models.SumSkyModel`"""
        return SumSkyModel(self.skymodels)


class SkyModel(object):
    """Sky model component.

    This model represents a factorised sky model.
    It has a `~gammapy.utils.modeling.ParameterList`
    combining the spatial and spectral parameters.

    TODO: add possibility to have a temporal model component also.

    Parameters
    ----------
    spatial_model : `~gammapy.image.models.SpatialModel`
        Spatial model (must be normalised to integrate to 1)
    spectral_model : `~gammapy.spectrum.models.SpectralModel`
        Spectral model
    name : str
        Model identifier
    """

    def __init__(self, spatial_model, spectral_model, name='SkyModel'):
        self.name = name
        self._spatial_model = spatial_model
        self._spectral_model = spectral_model
        self._parameters = ParameterList(
            spatial_model.parameters.parameters +
            spectral_model.parameters.parameters
        )

    @property
    def spatial_model(self):
        """`~gammapy.image.models.SkySpatialModel`"""
        return self._spatial_model

    @property
    def spectral_model(self):
        """`~gammapy.spectrum.models.SpectralModel`"""
        return self._spectral_model

    @property
    def parameters(self):
        """Parameters (`~gammapy.utils.modeling.ParameterList`)"""
        return self._parameters

    @parameters.setter
    def parameters(self, parameters):
        idx = len(self.spatial_model.parameters.parameters)
        self._spatial_model.parameters.parameters = parameters.parameters[:idx]
        self._spectral_model.parameters.parameters = parameters.parameters[idx:]

    def __repr__(self):
        fmt = '{}(spatial_model={!r}, spectral_model={!r})'
        return fmt.format(self.__class__.__name__,
                          self.spatial_model, self.spectral_model)

    def __str__(self):
        ss = '{}\n\n'.format(self.__class__.__name__)
        ss += 'spatial_model = {}\n\n'.format(self.spatial_model)
        ss += 'spectral_model = {}\n'.format(self.spectral_model)
        return ss

    def evaluate(self, lon, lat, energy):
        """Evaluate the model at given points.

        Return differential surface brightness cube.
        At the moment in units: ``cm-2 s-1 TeV-1 deg-2``

        Parameters
        ----------
        lon, lat : `~astropy.units.Quantity`
            Spatial coordinates
        energy : `~astropy.units.Quantity`
            Energy coordinate

        Returns
        -------
        value : `~astropy.units.Quantity`
            Model value at the given point.
        """
        val_spatial = self.spatial_model(lon, lat)
        val_spectral = self.spectral_model(energy)
        val_spectral = np.atleast_1d(val_spectral)[:, np.newaxis, np.newaxis]

        val = val_spatial * val_spectral

        return val.to('cm-2 s-1 TeV-1 deg-2')

    def copy(self):
        """A deep copy"""
        return copy.deepcopy(self)

    def __add__(self, skymodel):
        return CompoundSkyModel(self, skymodel, operator.add)

    def __radd__(self, model):
        return self.__add__(model)


class CompoundSkyModel(object):
    """Represents the algebraic combination of two
    `~gammapy.cube.models.SkyModel`

    Parameters
    ----------
    model1, model2 : `SkyModel`
        Two sky models
    operator : callable
        Binary operator to combine the models
    """

    def __init__(self, model1, model2, operator):
        self.model1 = model1
        self.model2 = model2
        self.operator = operator

    # TODO: Think about how to deal with covariance matrix
    @property
    def parameters(self):
        """Parameters (`~gammapy.utils.modeling.ParameterList`)"""
        return ParameterList(
            self.model1.parameters.parameters +
            self.model2.parameters.parameters
        )

    @parameters.setter
    def parameters(self, parameters):
        idx = len(self.model1.parameters.parameters)
        self.model1.parameters.parameters = parameters.parameters[:idx]
        self.model2.parameters.parameters = parameters.parameters[idx:]

    def __str__(self):
        ss = self.__class__.__name__
        ss += '\n    Component 1 : {}'.format(self.model1)
        ss += '\n    Component 2 : {}'.format(self.model2)
        ss += '\n    Operator : {}'.format(self.operator)
        return ss

    def evaluate(self, lon, lat, energy):
        """Evaluate the compound model at given points.

        Return differential surface brightness cube.
        At the moment in units: ``cm-2 s-1 TeV-1 deg-2``

        Parameters
        ----------
        lon, lat : `~astropy.units.Quantity`
            Spatial coordinates
        energy : `~astropy.units.Quantity`
            Energy coordinate

        Returns
        -------
        value : `~astropy.units.Quantity`
            Model value at the given point.
        """
        val1 = self.model1.evaluate(lon, lat, energy)
        val2 = self.model2.evaluate(lon, lat, energy)

        return self.operator(val1, val2)


class SumSkyModel(object):
    """Sum of independent `SkyModel` components.

    Not sure if we want this class, or only a + operator on SkyModel.
    If we keep it, then probably SkyModel should become an ABC
    and the current SkyModel renamed to SkyModelFactorised or something like that?

    Parameters
    ----------
    components : list
        List of SkyModel objects
    """

    def __init__(self, components):
        self.components = components
        pars = []
        for model in self.components:
            for p in model.parameters.parameters:
                pars.append(p)
        self._parameters = ParameterList(pars)

    @property
    def parameters(self):
        """Concatenated parameters.

        Currently no way to distinguish spectral and spatial.
        """
        return self._parameters

    @parameters.setter
    def parameters(self, parameters):
        idx = 0
        for component in self.components:
            n_par = len(component.parameters.parameters)
            component.parameters.parameters = parameters.parameters[idx:idx + n_par]
            idx += n_par

    def evaluate(self, lon, lat, energy):
        out = self.components[0].evaluate(lon, lat, energy)
        for component in self.components[1:]:
            out += component.evaluate(lon, lat, energy)
        return out


class SkyDiffuseCube(object):
    """Cube sky map template model (3D).

    This is for a 3D map with an energy axis.
    The map unit is assumed to be ``cm-2 s-1 MeV-1 sr-1``.
    Use `~gammapy.image.models.SkyDiffuseMap` for 2D maps.

    Parameters
    ----------
    map : `~gammapy.map.Map`
        Map template
    norm : float
        Norm parameter (multiplied with map values)
    meta : dict, optional
        Meta information, meta['filename'] will be used for serialization
    """

    def __init__(self, map, norm=1, meta=None):
        if len(map.geom.axes) != 1:
            raise ValueError('Need a map with an energy axis')

        axis = map.geom.axes[0]
        if axis.name != 'energy':
            raise ValueError('Need a map with axis of name "energy"')

        if axis.node_type != 'center':
            raise ValueError('Need a map with energy axis node_type="center"')

        self.map = map
        self._interp_opts = {'fill_value': 0, 'interp': 'linear'}
        self.parameters = ParameterList([
            Parameter('norm', norm),
        ])
        self.meta = {} if meta is None else meta

    @classmethod
    def read(cls, filename, **kwargs):
        """Read map from FITS file.

        Parameters
        ----------
        filename : str
            FITS image filename.
        """
        m = Map.read(filename, **kwargs)
        if m.unit == '':
            m.unit = 'cm-2 s-1 MeV-1 sr-1'
        return cls(m)

    def evaluate(self, lon, lat, energy):
        """Evaluate model."""
        energy = np.atleast_1d(energy)[:, np.newaxis, np.newaxis]
        energy = energy.to(self.map.geom.axes[0].unit).value

        coord = {
            'lon': lon.to('deg').value,
            'lat': lat.to('deg').value,
            'energy': energy,
        }
        val = self.map.interp_by_coord(coord, **self._interp_opts)
        norm = self.parameters['norm'].value
        return norm * val * u.Unit('cm-2 s-1 MeV-1 sr-1')
