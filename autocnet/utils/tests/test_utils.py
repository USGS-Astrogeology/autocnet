import unittest
import numpy as np
from .. import utils


class TestUtils(unittest.TestCase):

    def setUp(self):
        pass

    def test_checkbandnumbers(self):
        self.assertTrue(utils.checkbandnumbers([1, 2, 3, 4, 5], (2, 5, 1)))
        self.assertFalse(utils.checkbandnumbers([1, 2, 4], (1, 2, 3)))
        self.assertTrue(utils.checkbandnumbers([1.0, 2.0, 3.0], [1.0]))
        self.assertFalse(utils.checkbandnumbers(
            [-1.0, 2.0, 3.0], (1.0, 2.0, 3.0)))

    def test_getdeplaid(self):
        self.assertEqual(utils.checkdeplaid(95), 'night')
        self.assertEqual(utils.checkdeplaid(127.4), 'night')
        self.assertEqual(utils.checkdeplaid(180), 'night')
        self.assertEqual(utils.checkdeplaid(94.99), 'night')
        self.assertEqual(utils.checkdeplaid(90), 'night')
        self.assertEqual(utils.checkdeplaid(26.1), 'day')
        self.assertEqual(utils.checkdeplaid(84.99), 'day')
        self.assertEqual(utils.checkdeplaid(0), 'day')
        self.assertFalse(utils.checkdeplaid(-1.0))

    def test_checkmonotonic(self):
        self.assertTrue(utils.checkmonotonic(np.arange(10)))
        self.assertTrue(utils.checkmonotonic(range(10)))
        self.assertFalse(utils.checkmonotonic([1, 2, 4, 3]))
        self.assertFalse(utils.checkmonotonic([-2.0, 0.0, -3.0]))

        self.assertEqual(utils.checkmonotonic(np.arange(10), piecewise=True),
                         [True] * 10)
        self.assertEqual(utils.checkmonotonic(range(10), piecewise=True),
                         [True] * 10)
        self.assertEqual(utils.checkmonotonic([1, 2, 4, 3], piecewise=True),
                         [True, True, True, False])
        self.assertEqual(utils.checkmonotonic(
            [-2.0, 0.0, -3.0], piecewise=True), [True, True, False])

    def test_getnearest(self):
        iterable = range(10)
        idx, value = utils.getnearest(iterable, 3)
        self.assertEqual(idx, 3)

        idx, value = utils.getnearest(iterable, 8.32)
        self.assertEqual(idx, 8)

        idx, value = utils.getnearest(iterable, 8.5)
        self.assertEqual(idx, 8)

        idx, value = utils.getnearest(iterable, 8.51)
        self.assertEqual(idx, 9)

    def test_find_in_dict(self):
        d = {'a': 1,
             'b': 2,
             'c': {
                 'd': 3,
                 'e': 4,
                 'f': {
                     'g': 5,
                     'h': 6
                 }
             }
             }

        self.assertEqual(utils.find_in_dict(d, 'a'), 1)
        self.assertEqual(utils.find_in_dict(d, 'f'), {'g': 5, 'h': 6})
        self.assertEqual(utils.find_in_dict(d, 'e'), 4)

    def test_find_nested_in_dict(self):
        d = {'a': 1,
             'b': 2,
             'c': {
                 'd': 3,
                 'e': 4,
                 'f': {
                     'g': 5,
                     'h': 6
                 }
             }
             }

        self.assertEqual(utils.find_nested_in_dict(d, 'a'), 1)
        self.assertEqual(utils.find_nested_in_dict(d, ['c', 'f', 'g']), 5)

    def test_make_homogeneous(self):
        pts = np.arange(50).reshape(25, 2)
        pts = utils.make_homogeneous(pts)
        self.assertEqual(pts.shape, (25, 3))
        np.testing.assert_array_equal(pts[:, -1], np.ones(25))
