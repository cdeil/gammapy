# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Binning utility class and methods.

A `Binning` is defined as an array of bins,
where each bin as an index, a minimum and maximum,
and optionally belongs to a given group.

At the moment, this is used for energy binning.
In the future we will probably also use this for time binning.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from collections import OrderedDict
import numpy as np
from astropy.table import Table

__all__ = [
    'Binning'
]


class Bin(object):
    """Information on a single bin.

    Contains info from a row in the bin table.
    """
    UNDERFLOW_BIN_INDEX = -1
    OVERFLOW_BIN_INDEX = -2

    def __init__(self, bin_idx=None, group_idx=None,
                 min=None, max=None):
        self.bin_idx = bin_idx
        self.group_idx = group_idx
        self.min = min
        self.max = max

    def __repr__(self):
        return '{}({!r})'.format(
            self.__class__.__name__,
            self.to_dict(),
        )

    def __contains__(self, value):
        if (self.min <= value) and (value < self.max):
            return True
        else:
            return False

    def __eq__(self, other):
        return self.to_dict() == other.to_dict()

    def to_dict(self):
        """Convert to `~collections.OrderedDict`."""
        return OrderedDict([
            ('bin_idx', self.bin_idx),
            ('group_idx', self.group_idx),
            ('min', self.min),
            ('max', self.max),
        ])


# Subclassing is used here just for code re-use
class BinGroup(Bin):
    """Consecutive group of bins.

    Note: min and max ``bin_idx`` are inclusive!!!

    Contains info from a row in the groups table.
    """

    def __init__(self, group_idx=None,
                 bin_idx_min=None, bin_idx_max=None,
                 min=None, max=None):
        self.group_idx = group_idx
        self.bin_idx_min = bin_idx_min
        self.bin_idx_max = bin_idx_max
        self.min = min
        self.max = max

    def to_dict(self):
        """Convert to `~collections.OrderedDict`."""
        return OrderedDict([
            ('group_idx', self.group_idx),
            ('bin_idx_min', self.bin_idx_min),
            ('bin_idx_max', self.bin_idx_max),
            ('min', self.min),
            ('max', self.max),
        ])

    @property
    def bin_idx(self):
        """Array of bin indices (`~numpy.ndarray`).

        Can be used to index into Numpy arrays or tables.
        """
        return np.arange(*self.slice)

    @property
    def slice(self):
        """Slice (`slice`).

        Can be used to index into Numpy arrays or tables.
        """
        return slice(self.bin_idx_min, self.bin_idx_max + 1)


class Binning(object):
    """Binning table.

    Internally data is stored in ``bin_table``, but

    Note that only ordered consecutive binnings are supported.
    TODO: add checks and error handling.

    Parameters
    ----------
    bin_table : `~astropy.table.Table`
        Table with binning info
    """

    def __init__(self, bin_table):
        self.bin_table = bin_table

    @classmethod
    def from_bounds(cls, bounds, group_idx=None):
        bin_table = Table()
        bin_table['bin_idx'] = np.arange(len(bounds) - 1)
        bin_table['min'] = bounds[:-1]
        bin_table['max'] = bounds[1:]
        if group_idx is None:
            group_idx = np.arange(len(bounds) - 1)
        bin_table['group_idx'] = group_idx
        return cls(bin_table)

    @property
    def groups_table(self):
        """Table with one row per group (`~astropy.table.Table`)"""
        rows = []
        for group_idx in np.unique(self.bin_table['group_idx']):
            rows.append(self.group_dict(group_idx))

        names = ['group_idx', 'bin_idx_min', 'bin_idx_max', 'min', 'max']
        return Table(rows=rows, names=names)

    def find_bin(self, value):
        """Find bin that contains a given value."""
