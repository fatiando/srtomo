# Copyright (c) 2025 The SrTomo Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
These are the functions and classes that make up the SrTomo API.
"""

from ._jacobian import jacobian
from ._plot import plot_ray_paths
from ._utils import pair
from ._version import __version__

# Append a leading "v" to the generated version by setuptools_scm
__version__ = f"v{__version__}"
