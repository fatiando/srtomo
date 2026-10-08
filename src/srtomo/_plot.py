# Copyright (c) 2025 The SrTomo Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Plotting functions to help make some more complicated plots.
"""

import matplotlib.pyplot as plt
import numpy as np

from ._validation import to_arrays


def plot_ray_paths(sources, receivers, *, ax=None, **kwargs):
    """
    Plot the ray paths for every source and receiver pair.
    """
    sources = to_arrays(sources, ravel=True)
    receivers = to_arrays(receivers, ravel=True)
    if "linestyle" not in kwargs:
        kwargs["linestyle"] = "-"
    if "color" not in kwargs:
        kwargs["color"] = "k"
    if ax is None:
        ax = plt.gca()
    for p1, p2 in zip(np.transpose(sources), np.transpose(receivers), strict=True):
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], **kwargs)
