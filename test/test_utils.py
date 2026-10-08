# Copyright (c) 2025 The SrTomo Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Test our general utilities.
"""

import numpy.testing as npt

from srtomo._utils import pair


def test_pair():
    "Test the point pairing function against known results."
    sources = ([0, 1], [0, 0])
    receivers = ([1], [1])
    src, rec = pair(sources, receivers)
    expected_src = ([0, 1], [0, 0])
    expected_rec = ([1, 1], [1, 1])
    npt.assert_allclose(src, expected_src)
    npt.assert_allclose(rec, expected_rec)
