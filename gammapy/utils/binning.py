# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Binning utility class and methods.

A `Binning` is defined as an array of bins,
where each bin as an index, a minimum and maximum,
and optionally belongs to a given group.

At the moment, this is used for energy binning.
In the future we will probably also use this for time binning.

This is similar to
https://root.cern.ch/doc/master/classTAxis.html
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from collections import OrderedDict
import numpy as np
from astropy.table import Table

__all__ = [
    'Bin',
    'BinGroup',
    'Binning',
]


class Bin(object):
    """Information on a single bin.

    Contains info from a row in the bin table.
    """
    UNDERFLOW_BIN_INDEX = -1
    OVERFLOW_BIN_INDEX = -2

    _fields = ['bin_idx', 'group_idx', 'min', 'max']

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

    _fields = ['group_idx', 'bin_idx_min', 'bin_idx_max', 'min', 'max']

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
    """Axis binning and container and computations.

    This class can represent and do some computations with
    axis binnings that are ordered and consecutive, e.g.

    * Bin 0 from 10 to 11
    * Bin 1 from 11 to 15

    Usually such binnings are created from bounds arrays::

    >>> from gammapy.utils.binning import Binning
    >>> binning = Binning.from_bounds([10, 11, 15])

    The binning is stored in a `~astropy.table.Table` called
    ``bin_table`` with columns:

    * ``bin_idx`` - Bin index
    * ``group_idx`` - Bin group index
    * ``min`` - Bin left edge
    * ``max`` - Bin right edge

    This is mostly done for convenience, `~astropy.table.Table`
    is a nice way to store a few Numpy array with the same length.

    Parameters
    ----------
    bin_table : `~astropy.table.Table`
        Table with binning info

    Examples
    --------
    todo
    """

    def __init__(self, bin_table):
        self.bin_table = bin_table

    def __repr__(self):
        return '{}(n_bins={}, n_groups={})'.format(
            self.__class__.__name__,
            len(self.bin_table),
            len(self.groups_table),
        )

    @classmethod
    def from_bounds(cls, bounds, group_idx=None):
        """Create `Binning` from array of bounds.

        Parameters
        ----------
        bounds : array_like
            List or array of bin bounds
        """
        bin_table = Table()
        bin_table['bin_idx'] = np.arange(len(bounds) - 1)
        bin_table['min'] = bounds[:-1]
        bin_table['max'] = bounds[1:]
        if group_idx is None:
            group_idx = np.arange(len(bounds) - 1)
        bin_table['group_idx'] = group_idx
        return cls(bin_table)

    @classmethod
    def from_bin_list(cls, bins):
        """Create `Binning` from list of `Bin` objects."""
        rows = [bin.to_dict() for bin in bins]
        return Table(rows=rows, names=Bin._fields)

    def bin(self, row_idx):
        """Get `Bin` for a given bin index."""
        return Bin({colname: self.bin_table[row_idx][colname]
                    for colname in self.bin_table.colnames})

    @property
    def bin_list(self):
        """Get Python list of `Bin` for all bins."""
        return [self.get_bin(row_idx) for row_idx in range(len(self.bin_table))]

    @classmethod
    def from_groups_table(cls, groups_table):
        """Create `Binning` from a groups `~astropy.table.Table`."""
        bins = []
        for row in groups_table:
            group = BinGroup.from_row(row)
            bins.extend(group.to_bin_list())
        return cls.from_bin_list(bins)

    @property
    def groups_table(self):
        """Convert to groups table (`~astropy.table.Table`)."""
        rows = []
        for group_idx in np.unique(self.bin_table['group_idx']):
            rows.append(self.group_dict(group_idx))

        return Table(rows=rows, names=BinGroup._fields)

    # @property
    # def bin_bounds_array(self):
    #     return np.array(list(self.bin_table['min']).append(self.bin_table['max'][-1]))

    def regroup(self, other_binning):
        """Re-group this binning to some other given binning."""
        bin_table = self.bin_table.copy()
        for row in bin_table:
            group_idx = other_binning.find_bin_idx(row['min'])
            bin_table['group_idx'][row.index] = group_idx
        return self.__class__(bin_table)

    def find_bin_idx(self, value):
        """Find bin that contains a given value."""
        for bin in self.bin_list():
            if value in bin:
                return bin.bin_idx

        raise IndexError()
