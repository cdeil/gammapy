# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from astropy.utils.console import ProgressBar
from astropy.nddata.utils import PartialOverlapError
from astropy.coordinates import Angle
from ..maps import Map, WcsGeom
from .counts import fill_map_counts
from .exposure import make_map_exposure_true_energy
from .background import make_map_background_irf, _fov_background_norm

__all__ = [
    'MapMaker',
    'MapMakerObs',
]

log = logging.getLogger(__name__)


class MapMaker(object):
    """Make maps from IACT observations.

    Parameters
    ----------
    geom : `~gammapy.maps.WcsGeom`
        Reference image geometry
    offset_max : `~astropy.coordinates.Angle`
        Maximum offset angle
    exclusion_mask : `~gammapy.maps.Map`
        Exclusion mask
    cutout_mode : {'trim', 'strict'}, optional
        Options for making cutouts, see :func: `~gammapy.maps.WcsNDMap.make_cutout`
        Should be left to the default value 'trim'
        unless you want only fully contained observations to be added to the map
    """

    def __init__(self, geom, offset_max, exclusion_mask=None, cutout_mode="trim"):
        if not isinstance(geom, WcsGeom):
            raise ValueError('MapMaker only works with WcsGeom')

        if geom.is_image:
            raise ValueError('MapMaker only works with geom with an energy axis')

        self.geom = geom

        self.offset_max = Angle(offset_max)

        self.cutout_mode = cutout_mode

        # Start with zero-filled maps
        self.maps = {
            'counts': Map.from_geom(self.geom),
            'exposure': Map.from_geom(self.geom, unit="m2 s"),
            'background': Map.from_geom(self.geom),
        }

        # Some background estimation methods need an exclusion mask.
        if exclusion_mask is not None:
            self.maps['exclusion'] = exclusion_mask

    def run(self, obs_list):
        """
        Run MapMaker for a list of observations to create
        stacked counts, exposure and background maps

        Parameters
        --------------
        obs_list: `~gammapy.data.ObservationList`
            List of observations

        Returns
        -----------
        maps: dict of stacked counts, background and exposure maps.
        """
        for obs in ProgressBar(obs_list):
            # First make cutout of the global image
            try:
                # TODO: this is a hack. We should make cutout better.
                # See https://github.com/gammapy/gammapy/issues/1608
                cutout_map, cutout_slices = Map.from_geom(self.geom).make_cutout(
                    obs.pointing_radec, 2 * self.offset_max, mode=self.cutout_mode
                )
            except PartialOverlapError:
                # TODO: can we silently do the right thing here? Discuss
                log.warning("Observation {} not fully contained in target image. Skipping it.".format(obs.obs_id))
                continue

            log.info('Processing observation {}'.format(obs.obs_id))

            # Compute field of view mask on the cutout
            offset = cutout_map.geom.separation(obs.pointing_radec)
            fov_mask = offset >= self.offset_max

            # Only if there is an exclusion mask, make a cutout
            exclusion_mask = self.maps.get('exclusion', None)
            if exclusion_mask is not None:
                exclusion_mask, _ = exclusion_mask.make_cutout(
                    obs.pointing_radec, 2 * self.offset_max, mode=self.cutout_mode
                )

            map_maker_obs = MapMakerObs(
                obs=obs,
                geom=cutout_map.geom,
                fov_mask=fov_mask,
                exclusion_mask=exclusion_mask,
            )
            maps = map_maker_obs.run()

            # Stack maps from this observation into the total maps
            self.maps['counts'].data[cutout_slices] += maps['counts'].data
            self.maps['exposure'].data[cutout_slices] += maps['exposure'].quantity.to(self.maps['exposure'].unit).value
            self.maps['background'].data[cutout_slices] += maps['background'].data

        return self.maps


class MapMakerObs(object):
    """Make maps for a single IACT observation.

    Parameters
    ----------
    obs : `~gammapy.data.DataStoreObservation`
        Observation
    geom : `~gammapy.maps.WcsGeom`
        Reference image geometry
    fov_mask : `~numpy.ndarray`
        Mask to select pixels in field of view
    exclusion_mask : `~gammapy.maps.Map`
        Exclusion mask (used by some background estimators)
    """

    def __init__(self, obs, geom, fov_mask=None, exclusion_mask=None):
        self.obs = obs
        self.geom = geom
        self.fov_mask = fov_mask
        self.exclusion_mask = exclusion_mask

    def run(self):
        """tbd
        """
        counts = Map.from_geom(self.geom)
        fill_map_counts(counts, self.obs.events)
        if self.fov_mask is not None:
            counts.data[..., self.fov_mask] = 0

        exposure = make_map_exposure_true_energy(
            pointing=self.obs.pointing_radec,
            livetime=self.obs.observation_live_time_duration,
            aeff=self.obs.aeff,
            geom=self.geom,
        )
        if self.fov_mask is not None:
            exposure.data[..., self.fov_mask] = 0

        background = make_map_background_irf(
            pointing=self.obs.pointing_radec,
            livetime=self.obs.observation_live_time_duration,
            bkg=self.obs.bkg,
            geom=self.geom,
        )
        if self.fov_mask is not None:
            background.data[..., self.fov_mask] = 0

        # TODO: decide what background modeling options to support
        # This is not well tested or documented at the moment,
        # so for now take this out
        # background_scale = _fov_background_norm(
        #     acceptance_map=background,
        #     counts_map=counts,
        #     exclusion_mask=self.exclusion_mask,
        # )
        # if self.fov_mask is not None:
        #     background.data *= background_scale[:, None, None]

        return {
            'counts': counts,
            'exposure': exposure,
            'background': background,
        }
