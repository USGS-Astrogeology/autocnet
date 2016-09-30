import os
import sys
import unittest

import numpy as np

sys.path.append(os.path.abspath('..'))

from .. import utils


class TestUtils(unittest.TestCase):

    def test_normalize(self):
        r = np.random.RandomState(12345)
        coords = r.uniform(0,100,size=(100,2))

        normalizer = utils.normalize(coords)

        truth = np.array([[0.036613,  0, -1.995647],
                          [0, 0.036613, -1.862705],
                          [0, 0,  1]])

        np.testing.assert_array_almost_equal(normalizer, truth)
