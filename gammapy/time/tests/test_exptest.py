# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from ..exptest import run_exptest


def test_run_exptest():
    res = run_exptest()

    assert res == 42
