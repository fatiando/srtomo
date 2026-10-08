# Copyright (c) 2025 The SrTomo Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Functions to check and conform inputs and output.
"""

import numpy as np


def to_arrays(values, *, ravel=False):
    """
    Make sure all elements are numpy arrays.

    Conforms all elements of a tuple to numpy arrays and optionally ravel them.

    Parameters
    ----------
    values : tuple
        A tuple with the things to be converted to numpy arrays. Elements can be
        lists, tuples, arrays, or anything that can be turned into an array.
    ravel : bool
        If True, will ravel the arrays in the output, making them all 1D.

    Returns
    -------
    result : list
        List with the numpy arrays from the conversion.
    """
    result = [np.atleast_1d(x) for x in values]
    if ravel:
        result = [x.ravel() for x in result]
    return result
