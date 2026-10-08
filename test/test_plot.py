# Copyright (c) 2025 The SrTomo Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Test the plotting functions.

Compares code results against baseline images generated previously. To update
or make new baselines, run `make baseline`. New images will be placed in the
`baseline` top-level directory. Check the images and if they are correct, move
them to `test/baseline` and commit them.
"""

import matplotlib.pyplot as plt
import pytest

from srtomo._plot import plot_ray_paths


@pytest.mark.mpl_image_compare
def test_plot_ray_paths():
    "Check plotting ray paths"
    sources = ([0, 1], [0, 0])
    receivers = ([1, 1], [1, 1])
    fig, axes = plt.subplots(1, 2)
    plt.sca(axes[0])
    plot_ray_paths(sources, receivers)
    plot_ray_paths(sources, receivers, ax=axes[1], color="c", linestyle="--")
    for ax in axes:
        ax.set_xlim(-1, 2)
        ax.set_ylim(-1, 2)
    return fig
