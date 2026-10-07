# Copyright (c) 2025 The SrTomo Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Build the Jacobian matrix of the inversion.
"""

import numba
import numpy as np


@numba.jit(nopython=True)
def jacobian(sources, receivers, grid):
    """
    Calculate the Jacobian matrix of the tomography.
    """
    n_data = sources[0].size
    shape = grid[0].shape
    n_params = shape[0] * shape[1]
    xps = np.empty(6, dtype="float")
    yps = np.empty(6, dtype="float")
    cross = np.empty((6, 2), dtype="float")
    jacobian = np.zeros((n_data, n_params), dtype="float")
    cell_size = grid[0][0, 1] - grid[0][0, 0]
    for i in range(n_data):
        xs, ys = sources[0][i], sources[1][i]
        xr, yr = receivers[0][i], receivers[1][i]
        for l in range(shape[1]):
            x = grid[0][0, l]
            for m in range(shape[0]):
                y = grid[1][m, 0]
                x1, x2 = x - cell_size / 2, x + cell_size / 2
                y1, y2 = y - cell_size / 2, y + cell_size / 2
                maxx = max(xs, xr)
                maxy = max(ys, yr)
                minx = min(xs, xr)
                miny = min(ys, yr)
                # Check if the cell is in the rectangle with the ray path as a
                # diagonal. If not, then the ray doesn't go through the cell.
                if x2 < minx or x1 > maxx or y2 < miny or y1 > maxy:
                    continue
                # Now need to find the places where the ray intersects the cell
                # If the ray is vertical
                if (xr - xs) == 0:
                    xps[:] = xr
                    yps[0], yps[1], yps[2], yps[3] = yr, ys, y1, y2
                    intercept = 4
                # If the ray is horizontal
                elif (yr - ys) == 0:
                    xps[0], xps[1], xps[2], xps[3] = xr, xs, x1, x2
                    yps[:] = yr
                    intercept = 4
                else:
                    # Angular and linear coefficients of the ray
                    a_ray = (yr - ys) / (xr - xs)
                    b_ray = ys - a_ray * (xs)
                    # Add the src and rec locations so that the travel time of a
                    # src or rec inside a cell is accounted for
                    xps[0] = x1
                    xps[1] = x2
                    xps[2] = (y1 - b_ray) / a_ray
                    xps[3] = (y2 - b_ray) / a_ray
                    xps[4] = xs
                    xps[5] = xr
                    yps[0] = a_ray * x1 + b_ray
                    yps[1] = a_ray * x2 + b_ray
                    yps[2] = y1
                    yps[3] = y2
                    yps[4] = ys
                    yps[5] = yr
                    intercept = 6
                # Find out how many points are inside both the cell and the
                # rectangle with the ray path as a diagonal. Also remove the
                # duplicates
                crossings = _cross(
                    xps, yps, intercept, x1, x2, y1, y2, minx, maxx, miny, maxy, cross
                )
                if crossings > 2:
                    message = "Too many crossings"
                    raise ValueError(message)
                if crossings == 2:
                    distance = np.sqrt(
                        (cross[1, 0] - cross[0, 0]) ** 2
                        + (cross[1, 1] - cross[0, 1]) ** 2
                    )
                    jacobian[i, m * shape[1] + l] = distance
    return jacobian


@numba.jit(nopython=True)
def _cross(xps, yps, intercept, x1, x2, y1, y2, minx, maxx, miny, maxy, cross):
    """
    Find the crossings of the ray and cell boundaries.
    """
    k = 0
    for i in range(intercept):
        if (
            xps[i] <= x2
            and xps[i] >= x1
            and yps[i] <= y2
            and yps[i] >= y1
            and xps[i] <= maxx
            and xps[i] >= minx
            and yps[i] <= maxy
            and yps[i] >= miny
        ):
            duplicate = 0
            for j in range(k):
                if cross[j, 0] == xps[i] and cross[j, 1] == yps[i]:
                    duplicate = 1
                    break
            if duplicate:
                continue
            cross[k, 0] = xps[i]
            cross[k, 1] = yps[i]
            k += 1
    return k
