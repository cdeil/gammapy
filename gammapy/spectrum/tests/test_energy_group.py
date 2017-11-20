# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import astropy.units as u
from ..core import PHACountsSpectrum
from ..observation import SpectrumObservation
from ..energy_group import (
    SpectrumEnergyGroupMaker, SpectrumEnergyGroups,
    SpectrumEnergyGroup, SpectrumEnergyBin,
)


@pytest.fixture(scope='session')
def obs():
    """An example SpectrumObservation object for tests."""
    pha_ebounds = np.arange(1, 11) * u.TeV
    on_vector = PHACountsSpectrum(
        energy_lo=pha_ebounds[:-1],
        energy_hi=pha_ebounds[1:],
        data=np.zeros(len(pha_ebounds) - 1),
        meta=dict(EXPOSURE=99)
    )
    return SpectrumObservation(on_vector=on_vector)


class TestSpectrumEnergyGroupMaker:
    def test_str(self, obs):
        seg = SpectrumEnergyGroupMaker(obs=obs)
        ebounds = [1.25, 5.5, 7.5] * u.TeV
        seg.compute_groups_fixed(ebounds=ebounds)
        assert 'Number of groups: 4' in str(seg)

    def test_fixed(self, obs):
        ebounds = [1.25, 5.5, 7.5] * u.TeV
        seg = SpectrumEnergyGroupMaker(obs=obs)
        seg.compute_groups_fixed(ebounds=ebounds)
        t = seg.groups.to_group_table()
        assert_equal(t['energy_group_idx'], [0, 1, 2, 3])
        assert_equal(t['bin_idx_min'], [0, 1, 4, 6])
        assert_equal(t['bin_idx_max'], [0, 3, 5, 8])
        assert_equal(t['bin_type'], ['underflow', 'normal', 'normal', 'overflow'])

        ebounds = [1.0, 2.0, 5.0, 7.0, 10.0] * u.TeV
        assert_allclose(t['energy_min'], ebounds[:-1])
        assert_allclose(t['energy_max'], ebounds[1:])
        assert_equal(t['energy_group_n_bins'], [1, 3, 2, 3])

    def test_edges(self, obs):
        ebounds = [2, 5, 7] * u.TeV
        seg = SpectrumEnergyGroupMaker(obs=obs)
        seg.compute_groups_fixed(ebounds=ebounds)

        # We want thoses conditions verified
        t = seg.groups.to_group_table()
        assert_equal(len(t), 4)
        assert_equal(t['bin_type'], ['underflow', 'normal', 'normal', 'overflow'])
        assert_equal(t['bin_idx_min'], [0, 1, 4, 6])
        assert_equal(t['bin_idx_max'], [0, 3, 5, 8])
        assert_allclose(t['energy_min'], [1, 2, 5, 7] * u.TeV)
        assert_allclose(t['energy_max'], [2, 5, 7, 10] * u.TeV)
        assert_equal(t['energy_group_n_bins'], [1, 3, 2, 3])

    @pytest.mark.skip
    def test_adaptive(self, obs):
        seg = SpectrumEnergyGroupMaker(obs=obs)
        seg.compute_range_safe()
        seg.compute_groups_adaptive(quantity='sigma', threshold=2.0)

        # TODO: add asserts

@pytest.fixture()
def groups(obs):
    table = obs.stats_table()
    table['bin_idx'] = np.arange(len(table))
    table['energy_group_idx'] = table['bin_idx']
    return SpectrumEnergyGroups.from_total_table(table)


class TestSpectrumEnergyGroups:

    def test_str(self, groups):
        txt = str(groups)
        assert 'Number of groups: 9' in txt
        assert 'Bin range: (0, 8)' in txt
        assert 'Energy range: EnergyRange(min=1.0 TeV, max=10.0 TeV)' in txt

    def test_find_list_idx(self, groups):
        bin_idx_1TeV = groups.find_list_idx(energy=1 * u.TeV)
        bin_idx_5_9TeV = groups.find_list_idx(energy=5.9 * u.TeV)
        bin_idx_10TeV = groups.find_list_idx(energy=10 * u.TeV)
        assert_equal(bin_idx_1TeV, 0)
        assert_equal(bin_idx_5_9TeV, 4)
        assert_equal(bin_idx_10TeV, 8)

    def test_make_and_replace_merged_group(self, groups):
        # Merge first 4 bins
        groups.make_and_replace_merged_group(0, 3, 'underflow')
        assert_equal(groups[0].bin_type, 'underflow')
        assert_equal(groups[0].bin_idx_min, 0)
        assert_equal(groups[0].bin_idx_max, 3)
        assert_equal(groups[0].energy_group_idx, 0)

        # Flag 5th bin as normal
        groups.make_and_replace_merged_group(1, 1, 'normal')
        assert_equal(groups[1].bin_type, 'normal')
        assert_equal(groups[1].energy_group_idx, 1)

        # Merge last 4 bins
        groups.make_and_replace_merged_group(2, 5, 'overflow')
        assert_equal(groups[2].bin_type, 'overflow')
        assert_equal(groups[2].bin_idx_min, 5)
        assert_equal(groups[2].bin_idx_max, 8)
        assert_equal(groups[2].energy_group_idx, 2)
        assert_equal(len(groups), 3)

    def test_flag_and_merge_out_of_range(self, groups):
        ebounds = [2, 5, 7] * u.TeV
        groups.flag_and_merge_out_of_range(ebounds)

        t = groups.to_total_table()
        assert_equal(t['bin_type'], ['underflow', 'normal', 'normal', 'normal',
                                     'normal', 'normal', 'overflow', 'overflow', 'overflow'])
        assert_equal(t['energy_group_idx'], [0, 1, 2, 3, 4, 5, 6, 6, 6])

    def test_apply_energy_binning(self, groups):
        ebounds = [2, 5, 7] * u.TeV
        groups.apply_energy_binning(ebounds)

        t = groups.to_total_table()
        assert_equal(t['energy_group_idx'], [0, 1, 1, 1, 2, 2, 3, 4, 5])
        assert_equal(t['bin_idx'], [0, 1, 2, 3, 4, 5, 6, 7, 8])
        assert_equal(t['bin_type'], ['normal', 'normal', 'normal', 'normal',
                                     'normal', 'normal', 'normal', 'normal', 'normal'])


class TestSpectrumEnergyGroup:

    def test_init(self):
        g = SpectrumEnergyGroup()
        assert False


class TestSpectrumEnergyBin:

    def setup_class(self):
        self.b = SpectrumEnergyBin(idx=0, energy_min=3 * u.TeV, energy_max=10 * u.TeV)

    def test_str(self):
        assert str(self.b) == 'SpectrumEnergyBin(idx=0, energy_min=3.0 TeV, energy_max=10.0 TeV)'

    def test_contains(self):
        # Contains is left inclusive, right exclusive
        assert 2.9 * u.TeV not in self.b
        assert 3 * u.TeV in self.b
        assert 9.9 * u.TeV in self.b
        assert 10 * u.TeV not in self.b

    def test_energy_width(self):
        e = self.b.energy_width
        assert_allclose(e.value, 7)
        assert e.unit == 'TeV'

    def test_energy_log10_width(self):
        actual = self.b.energy_log10_width
        assert_allclose(actual, 0.5228787452803376)

    def test_energy_log_center(self):
        actual = self.b.energy_log_center
        assert_allclose(actual.value, 5.477225575051661)
        assert actual.unit == 'TeV'

    def test_list_from_energy_bounds(self):
        energy_bounds = [0.3, 1, 3, 10] * u.TeV
        b = SpectrumEnergyBin.list_from_energy_bounds(energy_bounds)
        assert len(b) == 3
