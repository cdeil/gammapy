# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
from astropy.table import Table
from ..binning import Bin, Binning


class TestBin:
    def test_bin_number(self):
        bin = Bin(bin_idx=42, min=3, max=5)
        assert 2.9 not in bin
        assert 3 in bin
        assert 4.9 in bin
        assert 5 not in bin

    def test_bin_quantity(self):
        # TODO: should it work with quantity values?
        pass


class TestBinning:
    @pytest.fixture(scope='session')
    def binning(self):
        """Example binning, used for most tests"""
        bounds = [10, 11, 12, 15, 17, 20]
        group_idx = [0, 0, 1, 1, 1]
        return Binning.from_bounds(bounds, group_idx)

    def test_bin_table(self, binning):
        t = binning.bin_table
        assert len(t) == 5

    def test_groups_table(self, binning):
        t = binning.groups_table
