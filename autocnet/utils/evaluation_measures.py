import numpy as np

"""
Code from: statsmodels - https://github.com/statsmodels/statsmodels

Released under a BSD-3 license:

Copyright (C) 2006, Jonathan E. Taylor
All rights reserved.

Copyright (c) 2006-2008 Scipy Developers.
All rights reserved.

Copyright (c) 2009-2012 Statsmodels Developers.
All rights reserved.


Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

  a. Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.
  b. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
  c. Neither the name of Statsmodels nor the names of its contributors
     may be used to endorse or promote products derived from this software
     without specific prior written permission.


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL STATSMODELS OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
"""


def mse(x1, x2, axis=0):
    """mean squared error
    Parameters
    ----------
    x1, x2 : array_like
       The performance measure depends on the difference between these two
       arrays.
    axis : int
       axis along which the summary statistic is calculated
    Returns
    -------
    mse : ndarray or float
       mean squared error along given axis.
    Notes
    -----
    If ``x1`` and ``x2`` have different shapes, then they need to broadcast.
    This uses ``numpy.asanyarray`` to convert the input. Whether this is the
    desired result or not depends on the array subclass, for example
    numpy matrices will silently produce an incorrect result.
    """
    x1 = np.asanyarray(x1)
    x2 = np.asanyarray(x2)
    return np.mean((x1 - x2)**2, axis=axis)


def rmse(x1, x2, axis=0):
    """root mean squared error
    Parameters
    ----------
    x1, x2 : array_like
       The performance measure depends on the difference between these two
       arrays.
    axis : int
       axis along which the summary statistic is calculated
    Returns
    -------
    rmse : ndarray or float
       root mean squared error along given axis.
    Notes
    -----
    If ``x1`` and ``x2`` have different shapes, then they need to broadcast.
    This uses ``numpy.asanyarray`` to convert the input. Whether this is the
    desired result or not depends on the array subclass, for example
    numpy matrices will silently produce an incorrect result.
    """
    x1 = np.asanyarray(x1)
    x2 = np.asanyarray(x2)
    return np.sqrt(mse(x1, x2, axis=axis))


def maxabs(x1, x2, axis=0):
    """maximum absolute error
    Parameters
    ----------
    x1, x2 : array_like
       The performance measure depends on the difference between these two
       arrays.
    axis : int
       axis along which the summary statistic is calculated
    Returns
    -------
    maxabs : ndarray or float
       maximum absolute difference along given axis.
    Notes
    -----
    If ``x1`` and ``x2`` have different shapes, then they need to broadcast.
    This uses ``numpy.asanyarray`` to convert the input. Whether this is the
    desired result or not depends on the array subclass.
    """
    x1 = np.asanyarray(x1)
    x2 = np.asanyarray(x2)
    return np.max(np.abs(x1 - x2), axis=axis)


def meanabs(x1, x2, axis=0):
    """mean absolute error
    Parameters
    ----------
    x1, x2 : array_like
       The performance measure depends on the difference between these two
       arrays.
    axis : int
       axis along which the summary statistic is calculated
    Returns
    -------
    meanabs : ndarray or float
       mean absolute difference along given axis.
    Notes
    -----
    If ``x1`` and ``x2`` have different shapes, then they need to broadcast.
    This uses ``numpy.asanyarray`` to convert the input. Whether this is the
    desired result or not depends on the array subclass.
    """
    x1 = np.asanyarray(x1)
    x2 = np.asanyarray(x2)
    return np.mean(np.abs(x1 - x2), axis=axis)


def medianabs(x1, x2, axis=0):
    """median absolute error
    Parameters
    ----------
    x1, x2 : array_like
       The performance measure depends on the difference between these two
       arrays.
    axis : int
       axis along which the summary statistic is calculated
    Returns
    -------
    medianabs : ndarray or float
       median absolute difference along given axis.
    Notes
    -----
    If ``x1`` and ``x2`` have different shapes, then they need to broadcast.
    This uses ``numpy.asanyarray`` to convert the input. Whether this is the
    desired result or not depends on the array subclass.
    """
    x1 = np.asanyarray(x1)
    x2 = np.asanyarray(x2)
    return np.median(np.abs(x1 - x2), axis=axis)


def bias(x1, x2, axis=0):
    """bias, mean error
    Parameters
    ----------
    x1, x2 : array_like
       The performance measure depends on the difference between these two
       arrays.
    axis : int
       axis along which the summary statistic is calculated
    Returns
    -------
    bias : ndarray or float
       bias, or mean difference along given axis.
    Notes
    -----
    If ``x1`` and ``x2`` have different shapes, then they need to broadcast.
    This uses ``numpy.asanyarray`` to convert the input. Whether this is the
    desired result or not depends on the array subclass.
    """
    x1 = np.asanyarray(x1)
    x2 = np.asanyarray(x2)
    return np.mean(x1 - x2, axis=axis)


def medianbias(x1, x2, axis=0):
    """median bias, median error
    Parameters
    ----------
    x1, x2 : array_like
       The performance measure depends on the difference between these two
       arrays.
    axis : int
       axis along which the summary statistic is calculated
    Returns
    -------
    medianbias : ndarray or float
       median bias, or median difference along given axis.
    Notes
    -----
    If ``x1`` and ``x2`` have different shapes, then they need to broadcast.
    This uses ``numpy.asanyarray`` to convert the input. Whether this is the
    desired result or not depends on the array subclass.
    """
    x1 = np.asanyarray(x1)
    x2 = np.asanyarray(x2)
    return np.median(x1 - x2, axis=axis)


def vare(x1, x2, ddof=0, axis=0):
    """variance of error
    Parameters
    ----------
    x1, x2 : array_like
       The performance measure depends on the difference between these two
       arrays.
    axis : int
       axis along which the summary statistic is calculated
    Returns
    -------
    vare : ndarray or float
       variance of difference along given axis.
    Notes
    -----
    If ``x1`` and ``x2`` have different shapes, then they need to broadcast.
    This uses ``numpy.asanyarray`` to convert the input. Whether this is the
    desired result or not depends on the array subclass.
    """
    x1 = np.asanyarray(x1)
    x2 = np.asanyarray(x2)
    return np.var(x1 - x2, ddof=ddof, axis=axis)


def stde(x1, x2, ddof=0, axis=0):
    """standard deviation of error
    Parameters
    ----------
    x1, x2 : array_like
       The performance measure depends on the difference between these two
       arrays.
    axis : int
       axis along which the summary statistic is calculated
    Returns
    -------
    stde : ndarray or float
       standard deviation of difference along given axis.
    Notes
    -----
    If ``x1`` and ``x2`` have different shapes, then they need to broadcast.
    This uses ``numpy.asanyarray`` to convert the input. Whether this is the
    desired result or not depends on the array subclass.
    """
    x1 = np.asanyarray(x1)
    x2 = np.asanyarray(x2)
    return np.std(x1 - x2, ddof=ddof, axis=axis)


def iqr(x1, x2, axis=0):
    """interquartile range of error
    rounded index, no interpolations
    this could use newer numpy function instead
    Parameters
    ----------
    x1, x2 : array_like
       The performance measure depends on the difference between these two
       arrays.
    axis : int
       axis along which the summary statistic is calculated
    Returns
    -------
    mse : ndarray or float
       mean squared error along given axis.
    Notes
    -----
    If ``x1`` and ``x2`` have different shapes, then they need to broadcast.
    This uses ``numpy.asarray`` to convert the input, in contrast to the other
    functions in this category.
    """
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    if axis is None:
        x1 = np.ravel(x1)
        x2 = np.ravel(x2)
        axis = 0
    xdiff = np.sort(x1 - x2)
    nobs = x1.shape[axis]
    idx = np.round((nobs - 1) * np.array([0.25, 0.75])).astype(int)
    sl = [slice(None)] * xdiff.ndim
    sl[axis] = idx
    iqr = np.diff(xdiff[sl], axis=axis)
    iqr = np.squeeze(iqr)  # drop reduced dimension
    return iqr
