# Copyright (c) 2025 The SrTomo Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Test that Jacobian matrix calculations are correct.
"""

import numpy as np
import numpy.testing as npt

from srtomo._jacobian import _cross, jacobian


def test_jacobian_simple():
    "Test the Jacobian using a simple model"
    spacing = 1
    diagonal = np.sqrt(2)
    grid = (
        [[0.5, 1.5], [0.5, 1.5]],
        [[0.5, 0.5], [1.5, 1.5]],
    )
    sources = ([0, 1, 2, 0, 0.5], [0, 0, 0, 1.5, 2])
    receivers = (
        [1, 2, 0, 2, 0.5],
        [1, 1, 2, 1.5, 0],
    )
    G = jacobian(sources, receivers, grid)
    expected = [
        [diagonal, 0, 0, 0],
        [0, diagonal, 0, 0],
        [0, diagonal, diagonal, 0],
        [0, 0, spacing, spacing],
        [spacing, 0, spacing, 0],
    ]
    npt.assert_allclose(G, expected)


def test_jacobian_empty():
    "Test the Jacobian using a simple model"
    sources = ([], [])
    receivers = (
        [1, 2, 0, 2, 0.5],
        [1, 1, 2, 1.5, 0],
    )
    grid = (
        [[0.5, 1.5], [0.5, 1.5]],
        [[0.5, 0.5], [1.5, 1.5]],
    )
    G = jacobian(([], []), ([], []), grid)
    npt.assert_allclose(G, np.zeros((0, 4)))



def test_crossings_vertical():
    "Check that finding vertical crossings works"
    x1, x2, y1, y2 = 0, 1, 0, 2
    xsrc, ysrc = 0.5, 0
    xrec, yrec = 0.5, 2
    xps = np.array([xsrc, xsrc, xsrc, xsrc])
    yps = np.array([yrec, ysrc, y1, y2])
    cross = np.zeros((6, 2), dtype="float")
    k = _cross(xps, yps, 4, x1, x2, y1, y2, xrec, xrec, ysrc, yrec, cross)
    assert k == 2, f"Number of crossings was {k} instead of 2"
    expected = np.array([[xrec, yrec], [xsrc, ysrc], [0, 0], [0, 0], [0, 0], [0, 0]])
    npt.assert_allclose(cross, expected)


def test_crossings_diagonal():
    "Check that finding diagonal crossings works"
    x1, x2, y1, y2 = 0, 1, 0, 2
    xsrc, ysrc = 0, 0
    xrec, yrec = 1, 2
    xps = np.array([x1, x2, xsrc, xrec, xsrc, xrec])
    yps = np.array([ysrc, yrec, y1, y2, ysrc, yrec])
    cross = np.zeros((6, 2), dtype="float")
    k = _cross(xps, yps, 6, x1, x2, y1, y2, xsrc, xrec, ysrc, yrec, cross)
    assert k == 2, f"Number of crossings was {k} instead of 2"
    expected = np.array([[xsrc, ysrc], [xrec, yrec], [0, 0], [0, 0], [0, 0], [0, 0]])
    npt.assert_allclose(cross, expected)


def test_crossings_off_diagonal():
    "Check that finding crossings works for a general case"
    x1, x2, y1, y2 = 0, 1, 0, 2
    xsrc, ysrc = 0, -1
    xrec, yrec = 1, 1
    xps = np.array([x1, x2, 0.5, 1, xsrc, xrec])
    yps = np.array([-1, 1, y1, y2, ysrc, yrec])
    cross = np.zeros((6, 2), dtype="float")
    k = _cross(xps, yps, 6, x1, x2, y1, y2, xsrc, xrec, ysrc, yrec, cross)
    assert k == 2, f"Number of crossings was {k} instead of 2"
    expected = np.array([[xrec, yrec], [0.5, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
    npt.assert_allclose(cross, expected)


def test_crossings_skip_no_crossings():
    ""
    x1, x2, y1, y2 = 0, 1, 0, 2
    xsrc, ysrc = 0, -1
    xrec, yrec = 1, 1
    xps = np.array([x1, x2, 0.5, 1, xsrc, xrec])
    yps = np.array([-1, 1, y1, y2, ysrc, yrec])
    cross = np.zeros((6, 2), dtype="float")
    k = _cross(xps, yps, 0, x1, x2, y1, y2, xsrc, xrec, ysrc, yrec, cross)
    assert k == 0, f"Number of crossings was {k} instead of 2"
