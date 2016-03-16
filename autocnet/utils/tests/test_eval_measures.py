import unittest
import numpy as np
from .. import evaluation_measures as em

from numpy.testing import assert_equal, assert_almost_equal

"""
Code modifid from: statsmodels - https://github.com/statsmodels/statsmodels

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


class TestEvalMeasures(unittest.TestCase):

    def setUp(self):
        pass

    def test_eval_measures(self):

        x = np.arange(20).reshape(4, 5)
        y = np.ones((4, 5))
        assert_equal(em.iqr(x, y), 5 * np.ones(5))
        assert_equal(em.iqr(x, y, axis=1), 2 * np.ones(4))
        assert_equal(em.iqr(x, y, axis=None), 9)

        assert_equal(em.mse(x, y),
                     np.array([73.5, 87.5, 103.5, 121.5, 141.5]))
        assert_equal(em.mse(x, y, axis=1),
                     np.array([3., 38., 123., 258.]))

        assert_almost_equal(em.rmse(x, y),
                            np.array([8.5732141, 9.35414347, 10.17349497,
                                      11.02270384, 11.89537725]))
        assert_almost_equal(em.rmse(x, y, axis=1),
                            np.array([1.73205081, 6.164414,
                                      11.09053651, 16.0623784]))

        assert_equal(em.maxabs(x, y),
                     np.array([14., 15., 16., 17., 18.]))
        assert_equal(em.maxabs(x, y, axis=1),
                     np.array([3., 8., 13., 18.]))

        assert_equal(em.meanabs(x, y),
                     np.array([7., 7.5, 8.5, 9.5, 10.5]))
        assert_equal(em.meanabs(x, y, axis=1),
                     np.array([1.4, 6., 11., 16.]))
        assert_equal(em.meanabs(x, y, axis=0),
                     np.array([7., 7.5, 8.5, 9.5, 10.5]))

        assert_equal(em.medianabs(x, y),
                     np.array([6.5, 7.5, 8.5, 9.5, 10.5]))
        assert_equal(em.medianabs(x, y, axis=1),
                     np.array([1., 6., 11., 16.]))

        assert_equal(em.bias(x, y),
                     np.array([6.5, 7.5, 8.5, 9.5, 10.5]))
        assert_equal(em.bias(x, y, axis=1),
                     np.array([1., 6., 11., 16.]))

        assert_equal(em.medianbias(x, y),
                     np.array([6.5, 7.5, 8.5, 9.5, 10.5]))
        assert_equal(em.medianbias(x, y, axis=1),
                     np.array([1., 6., 11., 16.]))

        assert_equal(em.vare(x, y),
                     np.array([31.25, 31.25, 31.25, 31.25, 31.25]))
        assert_equal(em.vare(x, y, axis=1),
                     np.array([2., 2., 2., 2.]))

        assert_almost_equal(em.stde(x, y), np.array(
            [5.59017, 5.59017, 5.59017, 5.59017, 5.59017]))
        assert_almost_equal(em.stde(x, y, axis=1), np.array(
            [1.4142136, 1.4142136, 1.4142136, 1.4142136]))
