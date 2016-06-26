# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.coordinates import Angle, SkyCoord
from ..extern.regions import CirclePixelRegion, PixCoord, CircleSkyRegion
from ..image import ExclusionMask

__all__ = [
    'find_reflected_regions',
]


def find_reflected_regions(region, center, exclusion_mask=None, angle_step='1 deg',
                           angle_dist_in='0 deg', angle_dist_reflected='0 deg'):
    """Find reflected regions outside the exclusion mask.

    The algorithm used to compute the reflected regions only works for circles
    and uses pixel coordinates internally.

    - Compute an `is_pos_outside` mask by dilating the exclusion mask by the circle radius
    - Step by `angle_step`, starting from `angle_dist_in`
      - Rotate region by `angle_step`
      - Lookup value


    is a stepping method that only works for circular regions,
    using pixel coordinates internally.
    First an

    Converts to pixel coordinates internally.

    Parameters
    ----------
    region : `~regions.CircleSkyRegion`
        Region
    center : `~astropy.coordinates.SkyCoord`
        Rotation point
    exclusion_mask : `~gammapy.image.ExclusionMask`
        Exclusion mask
    angle_step : `~astropy.coordinates.Angle`
        Rotation angle for each step
    angle_dist_in : `~astropy.coordinates.Angle`
        Minimal angular distance from input region
    angle_dist_reflected : `~astropy.coordinates.Angle`
        Minimal angular distance between to reflected regions

    Returns
    -------
    regions : list of `~regions.CircleSkyRegion`
        Reflected regions list
    """
    # This algorithm is implemented using a private class,
    # but exposed using a function in the public API.

    maker = _ReflectedRegionFinder(
        region=region,
        center=center,
        exclusion_mask=exclusion_mask,
    )
    maker.setup_rotation_angles(
        angle_step=angle_step,
        angle_dist_in=angle_dist_in,
        angle_dist_reflected=angle_dist_reflected,
    )
    maker.run()

    return maker.regions


class _ReflectedRegionFinder(object):
    """Reflected region find algorithm.

    Not exposed in the public API. See `find_reflected_regions` function.

    The class has the advantage of being able to implement the algorithm
    as a multi-step procedure (i.e. methods and properties) and using
    a few data members, so it's not necessary to hand things around and
    to debug the algorithm by accessing intermediate computed things
    after running the algorithm.
    """
    def __init__(self, region, center, exclusion_mask=None):
        # Make sure the inputs are OK
        self.region = CircleSkyRegion(region)
        self.center = SkyCoord(center)
        self.exclusion_mask = exclusion_mask

    def print_summary(self):
        """Print summary after running the algorithm.
        """

    angle_step = Angle(angle_step)
    angle_dist_reflected = Angle(angle_dist_reflected)
    angle_dist_in = Angle(angle_dist_in)
    fov_offset = region.center.separation(center)

    # If region overlaps center, there are no reflected regions
    if fov_offset <= region.radius:
        return []

    # Compute array of rotation angles at which to place the
    # rotated regions.
    angle_start = region.radius + angle_dist_in
    angle_stop = Angle('360 deg') - angle_dist_in
    if exclusion_mask is None:
        # In this case we can just compute the
        # reflected region positions directly
        angles = np.arange(
            start=angle_start,
            stop=angle_stop,
            step=2 * region.radius + angle_dist_reflected,
        )
    else:
        # In this case we use an array of test positions
        # and then filter it down.
        angles = np.arange(
            start=angle_start,
            stop=angle_stop,
            step=angle_step,
        )
        positions = _compute_rotated_position(
            pos=region.center,
            center=center,
            angles=angles,
        )
        valid_pos_mask = exclusion_mask.data
        mask = valid_pos_mask.lookup()

        # TODO: is this needed here?
        if sum(mask) == 0:
            return []

        # Filter list of test positions to those that correspond
        # to regions outside the exclusion mask
        angles = angles[mask]
        positions = positions[mask]

        # Filter list of test positions down further,
        # so that successive regions don't overlap
        mask = np.zeros_like(angles, dtype=bool)
        angles = []
        for angle, position in zip(angles, positions):
            if


    # Compute array of reflected region test positions




    # Convert coordinates to pixel and angles in radians.
    # This means we will have more efficient internal computations
    region = region.to_pixel(wcs=exclusion_mask.wcs)
    # TODO: check `origin` ... shouldn't it be zero? Add a test!
    center = PixCoord(*center.to_pixel(wcs=exclusion_mask.wcs, origin=1))
    angle_step = angle_step.radians
    angle_dist_reflected = angle_dist_reflected.radians
    angle_dist_in = angle_dist_in.radians

    # Compute offset in FOV and start and stop position angle for the algorithm
    dx_initial = region.center.x - center.x
    dy_initial = region.center.y - center.y
    angle_initial = np.arctan2(dx_initial, dy_initial)
    offset_in_fov = np.hypot(dx_initial, dy_initial)
    angle_start = 2 * region.radius / offset_in_fov + angle_dist_reflected
    angle_stop = angle_initial + 2 * np.pi - angle_start - angle_dist_in

    # import IPython; IPython.embed()

    # Compute list of reflected regions by stepping in rotation angle
    reflected_regions = []
    angle = angle_initial + angle_start + angle_dist_in
    while angle < angle_stop:
        test_pos = _compute_xy(center, offset_in_fov, angle)
        test_reg = CirclePixelRegion(test_pos, region.radius)
        if not _is_inside_exclusion(test_reg, exclusion_mask):
            reflected_regions.append(test_reg)
            angle = angle + angle_start
        else:
            angle = angle + angle_step

    # Convert reflected regions back to sky coordinates
    reflected_regions_sky = [_.to_sky(wcs=exclusion_mask.wcs) for _ in reflected_regions]

    return reflected_regions_sky


# TODO: replace by calculation using `astropy.coordinates`
def _compute_xy(pix_center, offset, angle):
    """Compute x, y position for a given position angle and offset
    """
    dx = offset * np.sin(angle)
    dy = offset * np.cos(angle)
    x = pix_center[0] + dx
    y = pix_center[1] + dy
    return x, y


# TODO :Copied from gammapy.region.PixCircleList (deleted), find better place
# Maybe `ExclusionMask.contains_region` ?
def _is_inside_exclusion(pixreg, exclusion):
    x, y = pixreg.center
    excl_dist = exclusion.distance_image
    val = excl_dist[np.round(y).astype(int), np.round(x).astype(int)]
    return val < pixreg.radius
