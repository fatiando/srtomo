# Copyright (c) 2025 The SrTomo Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
General utilities for the tomography.
"""

import numpy as np


def pair(sources, receivers):
    """
    Pair every source with every receiver.
    """
    xs, xr = [i.ravel() for i in np.meshgrid(sources[0], receivers[0])]
    ys, yr = [i.ravel() for i in np.meshgrid(sources[1], receivers[1])]
    return (xs, ys), (xr, yr)
